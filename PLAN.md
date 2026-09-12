# Indexter rewrite — implementation plan

## Context

`~/dev/indexter` (v0.1.2) indexes git repos with tree-sitter, embeds nodes with FastEmbed, and serves hybrid search from Qdrant over MCP. It works, but it carries weight it doesn't need: a Docker-hosted vector database, an async core where nothing is I/O-bound, a `repos.json` registry that can disagree with the collections it names, a CLI surface far wider than the agent workflow requires, and ~5,900 lines to do it.

This rewrite (`~/dev/indexter-rw`) keeps the purpose — **let an agent find code when the user can't name the symbol or module** — and adds the capability the old version lacked: **a code graph**. Vectors answer "what is this about"; the graph answers "what calls this", "what breaks if I change it", "how does a request reach the database". Neither alone covers the questions users actually ask.

Everything lands in one SQLite file per repo: embeddings, FTS5 keyword index, and graph. No server, no container, no daemon.

**Scope:** SQLite storage, lifted tree-sitter parsers, sentence-transformers embeddings, a FastMCP server with two tools, a skill, and a six-command CLI for setup only. **Nothing is lifted from the old repo except the parsers and the walker.**

---

## Architecture

```
repo files ──walk──▶ parse ──▶ ParsedNode + ParsedRef
                                    │
                        compose ────┼──── resolve (2-pass)
                          │         │          │
                       embed        │        edges
                          ▼         ▼          ▼
                  ┌──────────────────────────────────┐
                  │  SQLite: nodes · refs · edges     │
                  │  files · nodes_fts · vectors      │
                  └──────────────────────────────────┘
                                    ▲
                    search ─────────┘  (RRF: vector + FTS, then graph expansion)
                       ▲
                  FastMCP: search · neighbors
```

**Synchronous throughout.** SQLite is a local file, embedding is CPU/GPU-bound inside torch, walking is syscall-bound. The old async core bought nothing; FastMCP runs sync tool functions off the event loop.

---

## Settled decisions

| # | Decision |
|---|---|
| 1 | **Graph contents.** Nodes: `file` + symbols (`class`, `function`, `method`, `constant`, `interface`, `type_alias`, `enum`, `struct`, `trait`, `section`, data blocks) + `external_module`. Edges: `contains`, `imports`, `calls`, `inherits` — each with a line and a confidence. Imports/exports are edges, not nodes. No `references`/`type_of`/`instantiates` in v1. |
| 2 | **Embed composed summaries, not raw code.** MiniLM truncates at 256 word pieces and was trained on English. Summaries rank; **results always return real code**. |
| 3 | **sqlite-vec** for vectors (verified: v0.1.9, 3.1 ms KNN over 50k). No numpy fallback — uv's managed interpreters support extension loading; a missing extension fails loudly at startup. |
| 4 | **One database per repo, stored centrally**: `~/.local/share/indexter/<name>-<hash12>.db`, path derived deterministically from the canonical repo path. No registry file to drift. Per-repo *config* lives in the repo (`indexter.toml` / `[tool.indexter]`); global config in `~/.config/indexter/config.toml`. |
| 5 | **Sync on search**, always, no bypass. mtime+size check, then hash, then re-parse only what changed. |
| 6 | **Two MCP tools**: `search` (auto-expands one hop) and `neighbors`. Results carry bounded snippets plus `path:start-end`, so follow-ups are `sed`, not another round trip. |
| 7 | **RRF fusion** (top 50 vector + top 50 FTS, k=60), then one-hop expansion from the top 5. Graph-reached nodes go in a **separate "related" section**, never mixed into the ranking. |
| 8 | **Deterministic text node IDs**; `refs` are the durable source of truth and **edges are derived** from them. |
| 9 | **Resolution tiers** with a 5-candidate ambiguity cap and a builtin stoplist. |
| 10 | **sentence-transformers default** (lazy torch import), `fastembed` as the `onnx` extra. |
| 11 | **One global skill file**; no MCP auto-registration — the README documents it per client. |
| 12 | **Layout below**, synchronous core, existing conventions kept (uv, ruff, ty, pytest, ≥95% coverage, co-located tests). |
| 13 | **Build order M1–M7 below.** |

---

## Schema (`db/schema.sql`)

```sql
files(path PK, content_hash, language, size, mtime, indexed_at, node_count, errors)

nodes(rowid INTEGER PK,          -- unstable; vectors/FTS key off this
      id TEXT UNIQUE,            -- stable: src/auth/handlers.py::AuthHandler.login#method
      kind, name, name_words, qualified_name, file_path, language,
      start_line, end_line, start_byte, end_byte,
      signature, docstring, parent_id,
      embed_text, embed_hash, degree, updated_at)

refs(id INTEGER PK, from_node_id, raw_name, head, ref_kind,
     line, col, status, resolved_target_id, confidence, candidates JSON)

edges(id INTEGER PK, source TEXT, target TEXT, kind, line, confidence,
      UNIQUE(source, target, kind, IFNULL(line,-1)))

nodes_fts USING fts5(id, name, name_words, qualified_name,
                     docstring, signature, body)   -- bodies indexed here only

vectors USING vec0(node_rowid INTEGER PK,
                   kind TEXT, language TEXT,       -- filterable inside KNN
                   emb float[384])

project_metadata(key PK, value, updated_at)         -- repo_path, model, dim, schema_version
```

**Node bodies are not stored in `nodes`.** Snippets are read from disk by byte range (always current, since search syncs first); FTS5 keeps its own copy for BM25.

**Two ID spaces, deliberately.** Edges reference stable text IDs so re-indexing one file never invalidates another file's edges. Vectors and FTS key off integer rowids, rebuilt per file on sync.

**Filtering** (measured): `kind`/`language` filter *inside* the KNN (`kind IN (...)` works; `LIKE` is rejected — *"An illegal WHERE constraint was provided on a vec0 metadata column"*). Path prefixes become a rowid allowlist from `nodes` (2,000 IDs → 1.3 ms).

**Model changes** are detected via `project_metadata`; a dimension change rebuilds the `vectors` table (re-embed, not re-parse).

---

## Node IDs

```
src/auth/handlers.py::AuthHandler.login#method
src/auth/handlers.py::login#function~2        # ~N only for genuine duplicates, ordered by line
```

Format: `<relpath>::<scope path>.<name>#<kind>`. Stable across reformatting, line shifts and docstring edits; changes on rename, which is correct.

---

## Composer (`index/compose.py`)

Deterministic string building, no LLM. Ordered most-meaningful-first because truncation cuts the tail:

1. Label: kind + qualified name + **identifiers split into words** (`get_user_by_email` → "get user by email") + path words
2. Signature
3. Docstring summary/first paragraph — `Args:`/`Returns:`/`Raises:` blocks come last and drop first
4. Body prefix, blank lines collapsed, **string literals kept** (error messages, SQL, URLs are highly searchable)

Budget counted with the model's own tokenizer, final section cut at a token boundary. Per-kind variants: classes list member names instead of bodies; files list top-level symbols; markdown sections use heading path + prose (windowed if long); JSON/YAML/TOML use key path + slice.

Stored in `nodes.embed_text` with `embed_hash`, so re-embedding happens only when the text actually changes.

---

## Resolution (`index/resolve.py`)

Extraction writes every reference to `refs`; a second pass resolves them. First match wins:

| Tier | Rule | Confidence |
|---|---|---|
| 1 | `self`/`this`/`Self` → enclosing class, then bases via `inherits` | `exact` |
| 2 | Defined in same file, innermost enclosing scope | `exact` |
| 3 | Imported name or module-qualified call resolving to a repo file | `imported` |
| 4 | Name occurs exactly once in the repo | `unique_name` |
| 5 | 2–5 candidates → edge to each at reduced weight | `ambiguous` |
| — | >5 candidates → ref recorded with candidates, **no edges** (hairball guard) | — |
| — | No match → `failed`, retried after any later change | — |

**Filters:** per-language builtin/stdlib stoplist (dropped at extraction, not recorded as failures); external modules become `external::<name>` nodes — no embedding, in FTS by name, receive `imports` edges, so "what touches pydantic?" is one hop.

**Measured baseline** (34 files of the old repo, 1,079 call refs): 51.7% resolvable via imported/defined receiver, 17.1% bare known name, 11.5% `self`, 10.8% builtins (dropped), **8.9% genuinely ambiguous**. Expect ~80% high-confidence.

Import resolution is per-language and worth doing properly: Python relative/dotted, JS/TS extension + `index` resolution, Rust `crate::`/`mod`.

---

## Search (`search/`)

1. Sync changed files
2. Embed query (~5 ms warm); top 50 vector + top 50 FTS, filters applied per the rules above
3. RRF: `Σ 1/(60 + rank)`
4. Expand one hop from top 5 along `contains`/`calls` (both directions)/`inherits`/`imports`
5. **Hub damping**: skip expansion targets with degree > 40
6. `ambiguous` edges at reduced weight, never promoting alone
7. Roll-up: ≥2 methods of one class collapse into the class entry
8. Budget: 10 hits default, total character cap — degrade by returning fewer intact hits, never many truncated ones

**Result shape**: grouped by file; per hit — node ID, qualified name, kind, `path:start-end`, match reason (vector/keyword/graph), full signature + docstring, ≤40-line snippet (middle elided, not tail-cut), and up to 3 callers / 3 callees / container. Then a separate `related` section with the reason attached ("called by `authenticate`, which matched").

---

## Layout

```
src/indexter/
  config.py, paths.py
  db/        schema.sql, connection.py, queries.py
  walk.py
  parse/     base.py, python.py, javascript.py, typescript.py, rust.py,
             markdown.py, json.py, yaml.py, toml.py, html.py, css.py, chunk.py, models.py
  index/     compose.py, embed.py, resolve.py, sync.py
  search/    hybrid.py, expand.py, results.py
  mcp/       server.py, tools.py
  cli.py
  skill/SKILL.md
```

Tests co-located in `tests/` subdirectories, per existing convention.

**Dependencies:** `tree-sitter`, `tree-sitter-language-pack`, `sqlite-vec`, `sentence-transformers`, `fastmcp`, `typer`, `pydantic`, `pathspec`. One extra: `onnx` → `fastembed`. The old `core`/`cli`/`mcp`/`full` split goes away — it existed for Qdrant and Docker.

**CLI (setup only, no search):** `init`, `reindex`, `list`, `remove`, `mcp`, `skill`.

**MCP tools:** `search(query, repo?, kind?, language?, path?, limit?)` with `_meta: {"anthropic/alwaysLoad": true}`; `neighbors(node_id, direction, depth=1..3, limit)` via tool search. FastMCP `instructions=` frames hybrid+graph retrieval in two sentences.

---

## What to lift, and what to change

**Lift (the only code carried over):**
- `~/dev/indexter/src/indexter/parser/parsers/*.py` — all ten language parsers plus `chunk.py`
- `~/dev/indexter/src/indexter/walker/walker.py` — `IgnorePatternMatcher`, `BINARY_EXTENSIONS`, traversal and symlink guards

**Required parser changes:**
| File | Change |
|---|---|
| `parsers/base.py:140` | Compile `Query` once per parser instance, not per `parse()` call |
| `parsers/base.py:123-154` | Drop the `Document`/`NodeMetadata` coupling; emit `ParsedNode` + `ParsedRef` |
| `parsers/python.py:177` | Scope walk returns the **full ancestor path**, not just the nearest class (nested functions currently collide) |
| `parsers/javascript.py:218` | Same — callbacks and object-literal methods currently land at file scope |
| `parsers/rust.py:237` | Include the `trait` field of `impl_item`: `Foo<Display>.fmt` vs `Foo<Debug>.fmt` (currently collide) |
| all four code parsers | **New second query** capturing calls, imports and inheritance, taking the head identifier of nested attribute chains (`self.x.y()`, chained calls) |

Keep the existing duplicate-suppression logic (decorated definitions in Python, export-wrapped declarations in TypeScript) — it's correct.

**Walker changes:** make it synchronous (drop `anyio`), keep the filtering layers, keep `sha256(path:content)` hashing.

---

## Milestones

| # | Milestone | Done when |
|---|---|---|
| **M1** | Skeleton + DB: deps, config, paths, `schema.sql`, connection (pragmas, extension load, migrations), `list`/`remove` | Schema round-trips; missing sqlite-vec fails loudly and actionably |
| **M2** | Walk + parse: lift, sync, scope fixes, query caching, `ParsedNode`/`ParsedRef` | Per-language snapshot tests pass on the fixture repo |
| **M3** | Index pipeline: compose → embed → write nodes/refs/vectors/FTS; `init`, `reindex`, incremental sync | Indexes `~/dev/indexter`; re-run is a no-op; one edited file syncs well under a second |
| **M4** | Resolution + edges: tiers, external modules, stoplist, failed-ref retry | Fixture assertions pass; real-repo resolution lands near the measured ~80% high-confidence rate |
| **M5** | Search: RRF, filters, expansion, damping, roll-up, budget | `just eval` runs the question set; the Q2 composed-vs-raw A/B is decided here |
| **M6** | MCP + skill: two tools, instructions, `alwaysLoad`, `SKILL.md`, `indexter skill`, README client configs | Claude Code answers a "where is the code that…" question end-to-end |
| **M7** | Polish: CI matrix 3.11–3.13, ≥95% coverage, pre-commit, justfile, release | Green CI, publishable |

**M1–M4 carry the risk** — that's where the graph either works or doesn't. M5 onward is assembly over a database that already holds the right facts.

---

## Out of scope for v1

Watcher daemon; cross-repo search; languages beyond the current ten; LLM-written summaries; framework/route detection; `jina-embeddings-v2-base-code`. All additive later without schema changes, except the model swap, which needs a vector rebuild.

---

## Verification

**Fixture repo** (`tests/fixtures/`) — small multi-language tree with known imports, calls and inheritance, deliberately including the collision cases: Rust trait impls, Python nested functions, JS callbacks, same-name symbols in different files.

**Unit/integration:**
```bash
uv run --group test pytest --cov=indexter --cov-fail-under=95 --cov-report=term-missing
uv run --group dev ruff check --fix src/indexter && uv run --group dev ty check src/indexter
```

**Real-repo checks** (against `~/dev/indexter`, ~2,500 nodes):
- `indexter init ~/dev/indexter` completes; node/edge counts sane; full index embed time ~1–2 s
- `indexter reindex` immediately after is a no-op (zero re-embeds)
- Touch one file → next sync re-parses only that file
- Resolution report: ≥75% of call edges at `exact`/`imported`/`unique_name`; ambiguous ≤15%
- Known-answer spot checks: callers of `Walker.walk`, `inherits` from `BaseLanguageParser`, `imports` edges to `external::pydantic`

**Retrieval eval** (`just eval`, not in CI — ranking assertions are flaky): 15–20 natural-language questions with expected files, e.g. *"where do we decide which files to skip"* → `walker/walker.py`, *"how are decorated functions handled"* → `parsers/python.py`. Scored hit@5. Runs composed-vs-raw and model comparisons.

**End-to-end MCP:** register per README, then in Claude Code ask a question phrased with no symbol names and confirm `search` returns the right file with usable snippets, and `neighbors` walks from a returned node ID.

---

## Reference measurements (this machine)

| Measurement | Result |
|---|---|
| sqlite-vec KNN, 50k × 384 | 3.1 ms (insert 50k: 0.29 s) |
| sqlite-vec filtered KNN / rowid allowlist | 1.4 ms / 1.3 ms |
| sentence-transformers, warm | import 2.1 s, query 5 ms, batch 2,365/s (MPS) · 1,201/s (CPU) |
| fastembed, warm | import 0.2 s, query 4 ms, batch 260/s (thread/batch tuning changes nothing; CoreML is worse at 54/s) |
| Install footprint | ST 805 MB · fastembed 142 MB |
| Python call refs (old repo) | 80% high-confidence resolvable, 8.9% ambiguous, 10.8% builtins |
| Repo scale | largest local repo: 75 source files, ~2,000 definitions |
