<div align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/jdbadger/indexter/main/indexter-light.svg">
    <img src="https://raw.githubusercontent.com/jdbadger/indexter/main/indexter.png" alt="Indexter Logo">
  </picture>
</div>

<br>

<p align="center">
  <a href="https://github.com/jdbadger/indexter/actions/workflows/ci.yml"><img src="https://github.com/jdbadger/indexter/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://pypi.org/project/indexter/"><img src="https://img.shields.io/pypi/v/indexter" alt="PyPI"></a>
  <a href="https://pypi.org/project/indexter/"><img src="https://img.shields.io/pypi/pyversions/indexter" alt="Python versions"></a>
  <a href="https://github.com/jdbadger/indexter/blob/main/LICENSE"><img src="https://img.shields.io/pypi/l/indexter" alt="License"></a>
</p>

indexter indexes a codebase into a single local SQLite file: hybrid
(semantic + keyword) search over composed code summaries, plus a call/import/
inheritance graph, served to AI coding agents over two MCP tools,
`search` and `neighbors`. There is no server process to run, no vector
database to host, and no registry file: everything for a repository lives in
one database, keyed deterministically off its path.

## Supported Languages

Every file is parsed with tree-sitter into semantic units (functions, classes, config
tables, headings, and so on) rather than indexed as flat text.

| Language | Extensions | Semantic units extracted |
|---|---|---|
| Python | `.py` | Functions, methods, classes, module-level constants, docstrings |
| JavaScript | `.js`, `.jsx` | Functions, methods, classes, module-level constants, JSDoc comments |
| TypeScript | `.ts` | Functions, methods, classes, interfaces, type aliases, enums, module-level constants, TSDoc comments |
| Rust | `.rs` | Functions, methods, structs, traits, enums, type aliases, constants, doc comments (`///`, `//!`) |
| Markdown | `.md` | Sections, one per heading, named by their full heading path |
| HTML | `.html`, `.htm` | Sections: headings (`h1`–`h6`), tables and lists, scoped to their enclosing container |
| CSS | `.css` | Sections: rule sets and at-rules (`@media`, `@keyframes`, ...), scoped to their nesting |
| JSON | `.json` | Data nodes: objects and arrays, scoped to their key/index path |
| YAML | `.yaml`, `.yml` | Data nodes: block mappings and sequences, scoped to their key/index path |
| TOML | `.toml` | Data nodes: tables and key/value pairs, scoped to their dotted key path |
| *Anything else* | `*` | Fixed-size overlapping text chunks, so every file is searchable |

Python, JavaScript, TypeScript and Rust also resolve `calls`, `imports` and `inherits` edges
across files, forming the code graph `neighbors` walks. The rest produce containment structure
(`contains`) only, since there's nothing in JSON, a stylesheet, or a heading to call or import.

## Requirements

Python 3.11–3.13, run under a **uv-managed interpreter**. indexter stores
vectors with the [sqlite-vec](https://github.com/asg017/sqlite-vec) SQLite
extension, which needs a Python build whose `sqlite3` module supports loading
extensions; the interpreters `uv` installs support this, but the system
Python on macOS and some Linux distributions does not. Running indexter with
`uv run` (or a `uv`-installed tool, below) takes care of this automatically.

## Install

```bash
uv tool install --managed-python indexter
```

`--managed-python` makes sure the tool's own interpreter is one `uv`
installs, not whatever `python3` happens to resolve to on your system. The
interpreters uv manages support loading the `sqlite-vec` extension used for
vector storage, and a system Python often doesn't (see Requirements above).
This installs the `indexter` command on your `PATH`. Upgrade with
`uv tool upgrade indexter`; uninstall with `uv tool uninstall indexter`.

To install from a checkout instead (for a pre-release version, or to work on
indexter itself):

```bash
git clone https://github.com/jdbadger/indexter
uv tool install --managed-python ./indexter
```

Upgrading from a 0.1.x install? See the "Upgrading from 0.1" notes in
`CHANGELOG.md`.

## Getting started

Index a repository:

```bash
indexter init /path/to/repo
```

This walks the repository, parses it with tree-sitter, composes and embeds
each symbol, and writes the database. The first run downloads the embedding
model (about 90 MB for the default), so it can take a little longer; later
runs and re-syncs reuse the cached model. Re-index later with `indexter reindex
/path/to/repo` (add `--full` to delete and rebuild the database from
scratch instead of syncing changes).

While it works, `init` and `reindex` narrate their progress on stderr — loading
the model, indexing files, embedding — with each step resolving to a `✓` line,
and print their summary on stdout. The first-run download shows elapsed time
rather than a percentage, since the transfer size can't be known up front.
Narration appears only when stderr is a terminal, and only for steps slow
enough to notice, so a `reindex` with nothing to do prints just its summary.
Pass `--quiet` to turn it off, or `--progress` to turn it on when stderr is
redirected (a CI log, say). Colour follows `NO_COLOR`. stdout is identical
either way, so `indexter init . 2>/dev/null` is safe to parse.

Then register the MCP server with your agent (below) and install the skill
that teaches it when to use `search` and `neighbors`:

```bash
indexter skill --install
```

## Registering the MCP server

Every client runs the same command, `indexter mcp [--repo PATH]`. Without
`--repo`, the server resolves which repository a call targets from the
working directory it was started in, walking upward to the nearest indexed
ancestor, so a single, user-level registration works for every project a
workspace-based client opens. `--repo` pins one repository explicitly, for
clients with no workspace concept.

**Claude Code** (verified end-to-end):

```bash
claude mcp add --scope user indexter -- indexter mcp
```

**Claude Desktop** (no workspace: pin a repository):

```json
{
  "mcpServers": {
    "indexter": {
      "command": "indexter",
      "args": ["mcp", "--repo", "/path/to/repo"]
    }
  }
}
```

**VS Code** (`.vscode/mcp.json` in the workspace, so `cwd` resolves it):

```json
{
  "servers": {
    "indexter": {
      "command": "indexter",
      "args": ["mcp"],
      "cwd": "${workspaceFolder}"
    }
  }
}
```

**Cursor** (`.cursor/mcp.json`):

```json
{
  "mcpServers": {
    "indexter": {
      "command": "indexter",
      "args": ["mcp"]
    }
  }
}
```

Only the Claude Code configuration above has been exercised end-to-end; the
Claude Desktop, VS Code and Cursor configurations follow each client's
documented format but haven't been verified against a running client.

## The tools

- **`search(query, repo?, kind?, language?, path?, limit?)`**: hybrid
  semantic + keyword search. Returns matching code with each hit's closest
  callers, callees and containing scope. Always loaded in Claude Code.
- **`neighbors(node_id, repo?, direction?, edges?, depth?, limit?)`**: walks
  the call/import/inheritance/containment graph from a node ID returned by
  `search` (or a previous `neighbors` call): who calls this, what does this
  import, what inherits from this, 1–3 hops out. Loaded on demand.

Both tools sync the repository's index against the files on disk before
answering, so results always reflect the current working tree; there's no
separate reindex step to remember. Neither tool ever creates a database; an
unindexed repository comes back as an error naming `indexter init`.

## CLI commands

| Command | Description |
|---|---|
| `indexter init [PATH] [--quiet \| --progress]` | Create (or re-sync) a repository's index. Defaults to the current directory. Narrates progress on stderr; `--quiet` disables it, `--progress` forces it. |
| `indexter reindex [PATH] [--full] [--quiet \| --progress]` | Re-sync a previously initialized repository; `--full` rebuilds the database from scratch. Narrates like `init`. |
| `indexter list` | List indexed repositories, with node counts, embedding model and size. |
| `indexter remove TARGET [--yes]` | Remove an indexed repository's database (by repo path or database filename). Never touches the repository itself. |
| `indexter mcp [--repo PATH]` | Start the MCP server over stdio. |
| `indexter skill [--install] [--dir PATH] [--force]` | Print the packaged skill, or install it into an agent's skills directory. |
| `indexter --version` | Print the installed version and exit. |

## Configuration

Settings are layered: built-in defaults, then `~/.config/indexter/config.toml`
(global, applies to every repository), then `indexter.toml` or
`[tool.indexter]` in `pyproject.toml` at the repository root (per-repo,
overrides global). Unknown keys or wrong-typed values are rejected with an
error naming the key and the file it came from.

| Key | Default | Meaning |
|---|---|---|
| `embedding_model` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model name. |
| `embedding_dim` | `384` | Embedding vector dimension; must match the model. |
| `embedding_backend` | `sentence-transformers` | `sentence-transformers` or `fastembed`. |
| `embed_batch_size` | `32` | Texts embedded per batch during indexing. |
| `embed_max_tokens` | `256` | Truncation length for composed summaries. |
| `ignore_patterns` | `[]` | Extra gitignore-style patterns to skip while walking. |
| `max_file_size_bytes` | `1000000` | Files larger than this are skipped. |
| `search_limit` | `10` | Default number of results for `search`. |
| `snippet_max_lines` | `40` | Maximum lines shown per result snippet. |
| `search_max_chars` | `20000` | Character budget for a rendered response. |
| `chunk_size` | `1000` | Characters per chunk for oversized nodes. |
| `chunk_overlap` | `100` | Overlap between consecutive chunks. |

## File locations

- **Databases**: `$XDG_DATA_HOME/indexter/<slug>-<hash>.db` (defaults to
  `~/.local/share/indexter/`), one file per repository, named from the
  repository's canonical path. There is no registry: `indexter list` reads
  this directory directly.
- **Global config**: `$XDG_CONFIG_HOME/indexter/config.toml` (defaults to
  `~/.config/indexter/config.toml`).
- **Per-repo config**: `indexter.toml` or `[tool.indexter]` in
  `pyproject.toml`, at the repository root.
- **Installed skill**: `$CLAUDE_CONFIG_DIR/skills/indexter/SKILL.md`
  (defaults to `~/.claude/skills/indexter/SKILL.md`), or wherever
  `indexter skill --install --dir PATH` points.

## Contributing

See `CONTRIBUTING.md` for setting up a fork, the development recipes, and
the release process.
