## Why

After M5, search works and is measured, but only as a Python function: no agent can call it. M6 is where the rewrite meets its purpose — an agent in Claude Code (or any MCP client) asks "where is the code that…", gets real code back with its graph context, and can walk the graph from any returned node ID — so the tool surface, the way a repository is chosen, and the guidance an agent reads have to be decided here.

## What Changes

- Add `indexter mcp`: a FastMCP server over stdio exposing exactly two tools, with `instructions=` framing hybrid-plus-graph retrieval in two sentences. It resolves which repository a call targets (an explicit `repo` argument, else the `--repo` it was started with, else the nearest indexed ancestor of its working directory), reuses one loaded embedder per embedding configuration for the life of the process, warms the default embedder in the background at startup, serializes calls per repository, and turns every search, configuration, database and embedding failure into a tool error an agent can act on. Nothing but protocol messages is written to stdout.
- Add the `search` tool: `search(query, repo?, kind?, language?, path?, limit?)`, a thin wrapper over M5's `search`, returning its rendered text, marked `_meta: {"anthropic/alwaysLoad": true}` so Claude Code never defers it behind tool search.
- Add the `neighbors` tool and its core, `search/neighbors.py`: `neighbors(node_id, repo?, direction?, edges?, depth?, limit?)` syncs first (settled decision 5), then walks `calls`, `imports`, `inherits` and `contains` edges from a node ID, incoming, outgoing or both, 1–3 hops, without expanding through hubs, and renders a bounded, deterministic list with each neighbor's edge, confidence, depth and path. It carries a search hint instead of `alwaysLoad`, so Claude Code loads it on demand.
- Add `src/indexter/skill/SKILL.md` (packaged) and `indexter skill`: prints the skill, or `--install` writes it to Claude Code's user skills directory, refusing to overwrite a changed file without `--force`.
- Write the README: install, `indexter init`, registering the server per client (Claude Code, Claude Desktop, VS Code, Cursor), installing the skill, and configuration.
- Raise the `fastmcp` floor to the major version the server is written against.

## Capabilities

### New Capabilities
- `mcp-server`: The `indexter mcp` command and server — stdio transport, instructions, the two tools' parameters and metadata, repository resolution, embedder reuse and warm-up, per-repository serialization, error reporting, and a clean stdout.
- `graph-neighbors`: Walking the graph from a node ID — sync first, unknown-node errors, direction and edge-kind selection, depth 1–3 with hub and external-module stopping, deduplication and ordering, the limit and character budget, and the rendered text.
- `agent-skill`: The packaged `SKILL.md` — what it tells an agent — and the `indexter skill` command that prints or installs it.

### Modified Capabilities
<!-- None: `repo-management-cli` doesn't enumerate commands, so adding `mcp` and `skill` doesn't change its requirements; `hybrid-search` is wrapped, not changed. -->

## Impact

- **New code**: `mcp/__init__.py`, `mcp/server.py`, `mcp/tools.py`, `search/neighbors.py`, `skill/SKILL.md`, with co-located tests (in-memory FastMCP client over the M4 fixture repository with `FakeEmbedder`).
- **Changed code**: `cli.py` (`mcp`, `skill` commands); `search/types.py` (neighbor result types); `pyproject.toml` (`fastmcp` floor); `README.md` (written from empty).
- **Schema**: none. `neighbors` reads the existing `edges` and `nodes` tables.
- **Existing databases**: unchanged. Neither tool creates a database; an unindexed repository gets an error naming `indexter init`.
- **Dependencies**: none added (`fastmcp` is already a dependency; locked at 4.0.3).
- **Not touched**: parsing, composition, resolution, sync, ranking. No HTTP transport, no MCP auto-registration into client configs, no CLI search command.
- **User files**: `indexter skill --install` is the only command that writes outside the data directory, and only when asked.
