## 1. Setup

- [x] 1.1 Raise the `fastmcp` floor to `>=4.0` in `pyproject.toml` and confirm `uv lock` keeps 4.0.3
- [x] 1.2 Create `src/indexter/mcp/` (`__init__.py`, `server.py`, `tools.py`, `tests/`), `src/indexter/search/neighbors.py`, and `src/indexter/skill/` (`__init__.py`, `SKILL.md` placeholder)
- [x] 1.3 Add the frozen `Neighbor` and `NeighborsResponse` types to `search/types.py`, and the `NeighborsError` hierarchy (`InvalidArgument`, `NodeNotFound`) in `search/neighbors.py`

## 2. Neighbors core (`search/neighbors.py`)

- [x] 2.1 Validate `node_id`, `direction`, `edges`, `depth` (1–3) and `limit` (1–100, default 20) before syncing; errors name the parameter, value and valid values
- [x] 2.2 `neighbors_repo`: sync, look up the start node, and raise `NodeNotFound` with up to 5 same-name suggestions from the ID's file when that file is indexed
- [x] 2.3 Breadth-first walk: selected kinds and directions, first-seen at shallowest depth, recorded edge chosen by kind → confidence → via ID → line (orderings imported from `expand.py`), no walking through external modules, degree > 40 or ambiguous-only nodes (start node always walked), 1,000-node cap
- [x] 2.4 Order by depth → kind → confidence → ID, apply `limit`, then admit rendered items into `search_max_chars`; omitted count, lower-bound flag when capped
- [x] 2.5 Rendering: header (start node, ID, direction, edges, depth, shown/omitted), relation phrasing per kind and direction, `via` at depth > 1, ambiguous marking, no location for external modules, the no-neighbors line
- [x] 2.6 `neighbors(repo, …)`: derive the database path and raise the same `IndexNotFound` as search without creating a database, otherwise open and delegate
- [x] 2.7 Tests: every graph-neighbors spec scenario over hand-built node/edge sets (reusing the search test helpers), sync-before-read over a real fixture repository, and an inline snapshot of a rendered response over the M4 fixture repository

## 3. Server state and tool bodies (`mcp/tools.py`)

- [x] 3.1 `RepositoryNotFound(SearchError)` and repository resolution: `repo` argument (relative to working directory) → `--repo` default → working directory, then nearest indexed ancestor, never creating a database
- [x] 3.2 `ServerState`: default repository, working directory, embedder factory, embedder cache keyed by backend/model/dimension/batch size/token budget, one process-wide lock
- [x] 3.3 `run_search` and `run_neighbors`: under the lock resolve → `load_settings` → embedder → core → render; translate `SearchError`, `NeighborsError`, `ConfigError`, `IndexterDBError`, `EmbeddingError` into `ToolError`
- [x] 3.4 `warm_up(state)`: resolve a default repository without an argument; if found, under the lock load settings and embedder and embed one string; log failures to stderr; no-op when nothing resolves
- [x] 3.5 Tests: resolution scenarios (working directory, subdirectory upward, explicit wins, server default, nothing indexed creates nothing); embedder reuse and per-model separation; settings reloaded per call; error translation for each error family; warm-up success, failure and no-repository cases; two concurrent calls serialize

## 4. FastMCP server (`mcp/server.py`)

- [x] 4.1 `build_server(state)`: `FastMCP("indexter", instructions=…)` with the two-sentence instructions from design decision 7
- [x] 4.2 Register `search` with `Field` descriptions (valid kinds and languages interpolated from `FILTERABLE_KINDS` and `registered_languages()`), `meta={"anthropic/alwaysLoad": True}`, read-only/idempotent/closed-world annotations
- [x] 4.3 Register `neighbors` with `Literal` direction and edge kinds, bounded `depth`/`limit`, descriptions, `meta={"anthropic/searchHint": …}`, same annotations
- [x] 4.4 Lifespan hook that starts `warm_up` in a daemon thread without awaiting it; `run_server(default_repo)` runs stdio with `show_banner=False`
- [x] 4.5 Tests with an in-memory `fastmcp.Client` and `FakeEmbedder`: exactly two tools, annotations, `_meta` keys, schema descriptions list valid values, instructions name both tools, `search` and `neighbors` return the core's rendered text over the fixture repository, invalid input comes back as `isError`

## 5. CLI commands

- [x] 5.1 `indexter mcp [--repo PATH]`: reject a non-directory before starting, then `run_server`
- [x] 5.2 Write `skill/SKILL.md` per the agent-skill spec and design decision 8 (under 120 lines)
- [x] 5.3 `indexter skill [--install] [--dir PATH] [--force]`: print byte-for-byte; install to `$CLAUDE_CONFIG_DIR` or `~/.claude` under `skills/indexter/`; up-to-date, protected-edit and forced-overwrite handling; `--dir`/`--force` without `--install` rejected
- [x] 5.4 Tests: `mcp` with a missing `--repo` exits non-zero and never starts the server, and with a valid one calls `run_server` with it; every `indexter skill` scenario using `tmp_path` and a patched `CLAUDE_CONFIG_DIR`/home; skill frontmatter and required-terms checks
- [x] 5.5 Subprocess test: start `indexter mcp` in an empty temporary directory with isolated `XDG_DATA_HOME`/`XDG_CONFIG_HOME`, list tools over stdio with a FastMCP client, and assert the tool list and that stdout carried only protocol messages

## 6. README

- [x] 6.1 Write `README.md`: overview, requirements, install with `uv tool install` from a checkout, `indexter init`, CLI commands, configuration keys and file locations
- [x] 6.2 MCP registration for Claude Code (user scope), Claude Desktop (`--repo`), VS Code (`.vscode/mcp.json` with workspace `cwd`), Cursor (`.cursor/mcp.json`), noting only Claude Code is verified end-to-end; the two tools; installing the skill with `indexter skill --install`

## 7. Verification

- [x] 7.1 `uv run --group dev ruff check --fix src/indexter` clean
- [x] 7.2 `uv run --group dev ty check src/indexter` clean
- [x] 7.3 `uv run --group test pytest --cov=indexter --cov-fail-under=95 --cov-report=term-missing` green, and on Python 3.11, 3.12 and 3.13
- [x] 7.4 `uv build` and confirm the wheel contains `indexter/skill/SKILL.md` and no tests
- [x] 7.5 Real-repo check against `~/dev/indexter` with a stdio FastMCP client: warm-up completes, a warm `search` returns in well under 100 ms of server time, `neighbors` on a returned node ID with `direction="in"`, `edges=["calls"]` lists known callers (e.g. of `Walker.walk`), and `neighbors("external::pydantic", direction="in")` lists its importers
- [x] 7.6 Confirm Claude Code starts a user-scope stdio server with the project directory as its working directory (no `repo` argument needed), and that `search` is loaded without tool search while `neighbors` is deferred
- [x] 7.7 End-to-end (with the user's go-ahead, since it registers a server in their Claude Code config): register per the README, install the skill, ask Claude Code a "where is the code that…" question about `~/dev/indexter` phrased without symbol names, and confirm `search` returns the right file with usable snippets and `neighbors` walks from a returned node ID; record the question and outcome in design.md's Open Questions
