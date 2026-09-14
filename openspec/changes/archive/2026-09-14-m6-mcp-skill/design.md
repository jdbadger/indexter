## Context

After M5, `indexter.search.search(repo, query, settings, embedder, *, kind, language, path, limit)` syncs a repository, ranks, rolls up, budgets and expands, and `results.render(response)` turns the result into deterministic text sized for a language model (default cap 20,000 characters). It fails with typed `SearchError`s (`IndexNotFound`, `EmptyQuery`, `InvalidFilter`, `InvalidLimit`) and never creates a database. `search/expand.py` already holds the graph reads (`hit_context`, `expand`) and M5's design named them "reusable by M6's `neighbors`". The CLI has `init`, `reindex`, `list` and `remove`; `README.md` is empty; there is no MCP code and no skill.

The plan fixes the outline (settled decisions 5, 6, 11; the Layout and MCP-tools notes): two tools — `search(query, repo?, kind?, language?, path?, limit?)` with `_meta: {"anthropic/alwaysLoad": true}`, and `neighbors(node_id, direction, depth=1..3, limit)` found through tool search — FastMCP `instructions=` in two sentences, one global skill file, no MCP auto-registration (the README documents each client), and the CLI commands `mcp` and `skill`. M6 is done when Claude Code answers a "where is the code that…" question end-to-end.

Checked before writing this:

| Fact | Finding |
|---|---|
| FastMCP version locked | 4.0.3 (`pyproject.toml` still says `fastmcp>=0.4.1`) |
| Sync tool functions | `@server.tool(..., run_in_thread=True)` is the default: sync tools run in a worker thread, off the event loop |
| Tool metadata | `@server.tool(meta={...})` is emitted as the tool's `_meta`, merged with FastMCP's own `fastmcp` key |
| Claude Code's per-tool metadata (from the installed 2.1.269 binary) | reads `_meta["anthropic/alwaysLoad"] === true`, `_meta["anthropic/searchHint"]` (string), `_meta["anthropic/maxResultSizeChars"]` (number) |
| Server banner | `run(transport="stdio", show_banner=False)` suppresses it |
| Client roots | exposed only through the low-level async session; tool functions here are synchronous |
| Warm costs (plan's reference measurements) | sentence-transformers import 2.1 s, query 5 ms; no-op sync ~4 ms (M5) |
| Graph shape (`~/dev/indexter`, M5) | 8 of 2,679 nodes have degree > 40; p95 degree 6, p99 18 |

Constraints: synchronous core; sync on every tool call with no bypass; results carry `path:start-end` so follow-ups are reads, not round trips; tests co-located with ≥95% coverage and no network or model download; nothing but protocol on stdout under stdio.

## Goals / Non-Goals

**Goals:**
- `indexter mcp` that a user registers once per client and that picks the right repository without the agent having to name it.
- A `search` tool that is always loaded and adds nothing to M5's response but an agent-actionable error surface.
- A `neighbors` tool that answers "what calls this", "what does this import", "what inherits from this", and two- or three-hop walks, bounded like search.
- A skill that teaches an agent when to reach for `search` instead of grep, how to phrase a query, how to read the result, and when to call `neighbors`.
- A README a new user can follow from install to a working query in Claude Code.
- First search after server start not paying the model load when the repository is known at startup.

**Non-Goals:**
- HTTP/SSE transports, authentication, multi-client servers.
- Writing any client's MCP configuration (plan decision 11).
- MCP resources, prompts, structured tool output, or progress notifications.
- A CLI `search` or `neighbors` command — the CLI stays setup-only.
- Indexing from a tool call (`init` stays a setup step; M5 decision 1).
- Cross-repository search in one call.
- Publishing to PyPI, CI, pre-commit (M7).

## Decisions

### 1. Module layout: a thin server over plain functions

- `search/neighbors.py`: `neighbors_repo(conn, repo, node_id, settings, embedder, *, direction, edges, depth, limit)`, `neighbors(repo, …)` (opens the database like `search`), traversal, rendering, and `NeighborsError`s. It lives in `search/` beside `expand.py` because it is a graph read over the same tables, and `mcp/` stays free of SQL.
- `search/types.py` gains the frozen `Neighbor` and `NeighborsResponse` types.
- `mcp/tools.py`: `ServerState` (default repository, working directory, embedder factory, embedder cache, lock) and two synchronous functions, `run_search(state, …) -> str` and `run_neighbors(state, …) -> str`, that resolve the repository, load settings, get the embedder, call the core under the lock, render, and translate errors.
- `mcp/server.py`: `build_server(state) -> FastMCP` registers the two tools with their descriptions, parameter schemas, annotations and metadata, plus the lifespan warm-up; `run_server(default_repo)` builds state and runs stdio.
- `cli.py`: `mcp` and `skill` commands.
- `skill/SKILL.md` (with `skill/__init__.py` so `importlib.resources` finds it, as `db/schema.sql` is found today).

*Why tool bodies outside FastMCP:* they are tested as plain functions over the M4 fixture repository with `FakeEmbedder`; a handful of in-memory `fastmcp.Client` tests then check registration, schemas, metadata and error transport without re-testing search.

### 2. Which repository a call targets

Resolution order, first that applies:

1. the tool call's `repo` argument (absolute, or relative to the server's working directory);
2. the `--repo` path `indexter mcp` was started with;
3. the server process's working directory.

The chosen path is then walked **upward** (itself first) to the nearest directory whose derived database file exists (`paths.db_path`), and that directory is the repository. If none exists, the call fails with an error naming the starting path and the command to fix it (`indexter init <path>`); nothing is created.

*Why the working directory:* Claude Code, VS Code and Cursor start stdio servers in the workspace, so a user-scope registration works for every project without per-project configuration. *Why walk upward:* a session started in `src/` or an agent passing a subdirectory still means the enclosing repository; the walk costs one path hash and `stat` per level. *Why not MCP roots:* they arrive through an async request on the session, while tools are synchronous and the working directory already answers the question for the clients that matter; roots are an additive later change (Open Questions). *Why `--repo`:* Claude Desktop has no workspace, so its configuration pins one.

Settings are loaded with `load_settings(repo)` on every call — three small TOML reads — so an edited `indexter.toml` applies without restarting the server.

### 3. One embedder per embedding configuration, one lock, background warm-up

`ServerState` caches embedders keyed by `(embedding_backend, embedding_model, embedding_dim, embed_batch_size, embed_max_tokens)`, created with `make_embedder` on first use and kept for the process. Repositories with the same embedding settings share one loaded model.

Every tool call runs its resolution-to-render body under **one process-wide lock**. At server start the lifespan hook starts a daemon thread that, when a default repository resolves (decision 2, steps 2–3), takes the lock, loads that repository's settings and embedder, and embeds one short string. Failures in warm-up are written to stderr and otherwise ignored — the first tool call reports them properly. With no resolvable repository at start, there is no warm-up.

*Why one lock and not per repository:* Claude Code issues parallel tool calls; two concurrent syncs of one repository would race for SQLite's writer lock, and concurrent `encode` calls on one torch model aren't something to rely on. A warm search is tens of milliseconds, so serialization costs nothing measurable; per-repository locks plus per-embedder locks would need a lock order and buy only cross-repository parallelism, which no workflow here uses. A call that arrives during warm-up waits for the load it would otherwise have done itself.

*Why warm-up in a thread:* the client's `initialize` must answer immediately; a 2-second import in the handshake would look like a hung server.

### 4. The `search` tool

Parameters, all but `query` optional: `query: str`; `repo: str`; `kind: str | list[str]`; `language: str | list[str]`; `path: str`; `limit: int` (1–50). Each has a one-sentence `Field(description=…)` naming the valid values where there is a closed set (the filterable kinds and registered languages are interpolated from the code, not typed twice). The tool returns M5's `render(response)` text unchanged.

Metadata `{"anthropic/alwaysLoad": True}`; annotations `readOnlyHint=True`, `idempotentHint=True`, `openWorldHint=False` — it updates indexter's own database but never the repository or anything outside the machine.

*Why unchanged text:* M5 sized and measured the rendering for exactly this consumer; a second format would need its own budget.

*Why `alwaysLoad` only on `search`:* search is the entry point an agent must see to know indexter exists; `neighbors` is only useful after a search has returned node IDs, so deferring it behind tool search costs one lookup in the sessions that use it and nothing in those that don't.

### 5. The `neighbors` tool and traversal

Parameters: `node_id: str`; `repo: str`; `direction: "in" | "out" | "both"` (default `both`); `edges: list["calls" | "imports" | "inherits" | "contains"]` (default all four); `depth: int` 1–3 (default 1); `limit: int` 1–100 (default 20). Metadata `{"anthropic/searchHint": "code graph: callers, callees, imports, inheritance of a node from indexter search"}`; same annotations as `search`.

`edges` is an addition to the plan's signature: "what calls this" is the most common follow-up, and without a kind filter a class's answer is buried under its `contains` children.

**Validation** (before syncing, like search): blank `node_id`, unknown `direction`, unknown edge kind, `depth` or `limit` out of range → `InvalidArgument` naming the parameter, the value and the valid values.

**Sync, then look up** the node. An unknown ID → `NodeNotFound`, whose message says the ID may be stale and to search again, and — when the ID's `<path>::` part names an indexed file — lists up to 5 current IDs in that file with the same `name` (a renamed kind or a new `~N` suffix is the usual cause).

**Traversal**, breadth-first from the start node. At depth `d`, for each frontier node, follow the selected edge kinds in the selected directions. A node not yet seen (the start node counts as seen) is recorded at depth `d` with the edge that reached it and the frontier node it came from (`via`). When several edges reach it at the same depth, the recorded one is the first by (edge kind order `calls`, `inherits`, `imports`, `contains`; confidence order `exact`, `imported`, `unique_name`, `ambiguous`; `via` ID; line). The next frontier is every node recorded at depth `d` **except**: external modules, nodes with degree > 40, and nodes reached only through `ambiguous` edges — they are listed but not walked through. The start node is always walked, whatever its degree or kind, so `neighbors("external::pydantic", direction="in")` lists every importer. Traversal stops at `depth`, or once 1,000 nodes are recorded (then the total is reported as a lower bound).

**Result order**: depth, then edge kind order, then confidence order, then node ID. The first `limit` are returned; the rest are counted as omitted. The rendered response is also held to `search_max_chars`, admitting items in order and stopping at the first that doesn't fit, counted with the omitted.

**Rendering** (illustrative; the layout is pinned by a snapshot test):

```
neighbors of Walker._should_skip — method — src/indexter/walker/walker.py:120-158 (direction=both, edges=all, depth=1): 3 shown

id: src/indexter/walker/walker.py::Walker._should_skip#method

- calls Walker._is_binary_file — method — src/indexter/walker/walker.py:60-70 — exact, line 131
  id: src/indexter/walker/walker.py::Walker._is_binary_file#method
- called by Walker.walk — method — src/indexter/walker/walker.py:170-230 — exact, line 188
  id: src/indexter/walker/walker.py::Walker.walk#method
- contained in Walker — class — src/indexter/walker/walker.py:30-240
  id: src/indexter/walker/walker.py::Walker#class
```

Relations read from the `via` node to the neighbor: `calls` / `called by`, `inherits from` / `inherited by`, `imports` / `imported by`, `contains` / `contained in`. Items at depth > 1 add `via <qualified name>`; `ambiguous` edges are labeled as in search; external modules have no location. No snippets — `path:start-end` is enough to read one (settled decision 6).

*Why not reuse `expand` directly:* expansion scores and trims a one-hop neighborhood for ranking; `neighbors` is an exhaustive, ordered walk the caller steers. They share the edge-kind and confidence orderings and the hub threshold, imported from `expand.py` rather than restated.

*Why stop at hubs but list them:* a hub is a legitimate neighbor ("this calls `Document`"), but walking through a 300-degree node turns depth 2 into the whole repository.

*Why not walk through ambiguous-only nodes:* up to five candidates per ambiguous call compound per hop; the same "never promote alone" rule M5 uses for `related`.

### 6. Errors reach the agent as tool errors

`run_search` and `run_neighbors` catch `SearchError`, `NeighborsError`, `ConfigError`, `IndexterDBError` and `EmbeddingError` and raise FastMCP's `ToolError` with the exception's message, so the client receives `isError: true` with text an agent can act on (`run \`indexter init /path\``, "valid values: …"). A repository-resolution failure is a `SearchError` subclass (`RepositoryNotFound`) so both tools share it. Anything else propagates to FastMCP's default handling — it is a bug, not an expected outcome.

### 7. `indexter mcp`

`indexter mcp [--repo PATH]`. `--repo`, when given, must be an existing directory, checked before the server starts (exit 1 otherwise); it need not be indexed yet. The command runs the stdio transport with the banner off.

Stdout carries only protocol messages: no `typer.echo` after startup, FastMCP's logging on stderr, and sync and embedding code already print nothing. A subprocess test starts `indexter mcp` in an empty directory (no warm-up, no model), lists tools over stdio, and fails on any non-protocol stdout.

Instructions (the two sentences):

> indexter finds code in this repository by meaning and keywords together, and returns real code with each hit's callers, callees and related code from its call, import and inheritance graph. Use `search` with a plain-language description when you don't know the file or symbol name, then `neighbors` on a returned node ID to walk its graph further.

### 8. The skill and `indexter skill`

`SKILL.md` has frontmatter `name: indexter` and a `description` that says when to use it (finding code by behavior or concept, tracing callers/callees/imports, before grepping for a name you're guessing). The body covers: search before grep when the name is unknown; phrasing a query as the behavior in plain words; filters and when they help; reading a result (entries vs `related`, `path:start-end` then read the lines, node IDs); `neighbors` for callers, callees, importers and subclasses and when to raise `depth`; that every call syncs first, so results reflect the files as they are on disk; and what to do with a "no index" error (ask the user before running `indexter init`, which downloads a model and takes a while). It names tools as `search`/`neighbors` and notes Claude Code's `mcp__indexter__search` form. Target under 120 lines.

`indexter skill` prints the file. `indexter skill --install` writes it to `<config>/skills/indexter/SKILL.md`, `<config>` being `$CLAUDE_CONFIG_DIR` or `~/.claude`; `--dir PATH` replaces `<config>/skills/indexter` for other agents' skill directories. An existing identical file is reported as up to date; a different one is left alone with exit 1 unless `--force`. `--force` and `--dir` without `--install` are usage errors.

*Why print by default:* installation writes into another tool's configuration; the user should ask for that explicitly, and printing works for every agent.

### 9. README

Written from empty, covering only what exists after M6: what indexter does (one paragraph); requirements (Python 3.11–3.13 with SQLite extension loading, which uv-managed interpreters have); install with `uv tool install` from a checkout (PyPI is M7); `indexter init`; registering the server — Claude Code (`claude mcp add --scope user indexter -- indexter mcp`), Claude Desktop (`--repo` pinned), VS Code (`.vscode/mcp.json` with `cwd` set to the workspace), Cursor (`.cursor/mcp.json`); installing the skill; the two tools; configuration keys and file locations; the CLI commands.

### 10. Dependency floor

`fastmcp>=4.0`: the tool decorator's `meta`, `annotations` and `run_in_thread`, the lifespan hook, and `show_banner` are what the server uses, and the lock already pins 4.0.3. No other dependency changes.

## Risks / Trade-offs

- **A client that doesn't start stdio servers in the workspace** → resolution falls back to the server's own working directory, which may hold no index; the error names the path it tried and both fixes (`repo` argument, `--repo`). The README gives Claude Desktop a pinned `--repo`.
- **Only Claude Code is verified end-to-end** → Claude Desktop, VS Code and Cursor configurations follow each client's documented format but aren't exercised here; the README says so.
- **Claude Code's `_meta` keys are read from its current binary, not a published contract** → if they change, `search` is deferred behind tool search like any other tool: slower to discover, not broken.
- **One process-wide lock** → a call that triggers a long re-index (a branch switch) blocks a concurrent call to another repository; acceptable for one agent session per server process.
- **The first search after start still pays the model load when no repository resolves at startup** → the same ~2 s the CLI pays; tool timeouts in clients are far longer.
- **Parallel tool calls from Claude Code all sync** → the second finds nothing to do (~4 ms); correctness over a bypass, per settled decision 5.
- **`neighbors` from a large file at depth 3** → bounded by hub stopping, the 1,000-node cap, `limit` and the character budget; the omitted count tells the agent to narrow `edges` or `direction`.
- **Node IDs go stale after renames** → `NodeNotFound` suggests current IDs in the same file and says to search again.
- **The skill can drift from the tools' behavior** → a test asserts that the skill names both tools and every `neighbors` parameter, and that `indexter skill` prints the packaged file byte-for-byte.
- **Warm-up runs in a daemon thread with the lock held** → if model loading hangs (network), tool calls wait with it; the same call without warm-up would hang identically.

## Migration Plan

No schema change and no re-index. New commands and a new package; existing CLI commands, databases and configuration are untouched. Rollback is removing `mcp/`, `search/neighbors.py`, `skill/` and the two commands.

## Open Questions

- **MCP roots** — whether any target client starts servers outside the workspace often enough to justify reading `roots/list` (async, per session) as a resolution step before the working directory. Revisit if the end-to-end check or user reports show wrong-repository resolution.
- **Should `ambiguous` method fan-out collapse to a shared base method** (carried from M4/M5) — `neighbors` over the parsers during verification shows how noisy ambiguous `calls` edges are in practice; the collapse itself would be a resolution change.
- **`anthropic/maxResultSizeChars`** — not set: the 20,000-character default is well under Claude Code's default limit for tool output. Set it only if users raise `search_max_chars` past what the client accepts.
- **End-to-end result (task 7.7)** — registered `indexter mcp` at Claude Code user scope (no `--repo`) and installed the skill, then, from a one-shot Claude Code session started with `~/dev/indexter` as its working directory, asked: *"where in this codebase is the logic that decides whether a file should be skipped while walking a directory tree (e.g. because it matches a gitignore-style pattern)?"* (phrased without naming any symbol). It answered with a single `mcp__indexter__search` call — no `--repo` argument, no prior tool-search step — correctly naming `IgnorePatternMatcher.should_ignore`, `Walker._build_matcher` and `Walker.walk`/`_walk_recursive` in `src/indexter/walker/walker.py`, matching a manual check of the same query. `neighbors` wasn't needed for this question. The registration was removed afterward.
