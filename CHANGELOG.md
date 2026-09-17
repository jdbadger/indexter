# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Security

- The walker checked symlinked directories for repository containment but not symlinked
  files, and neither checked whether a symlink's target was itself an ignored path. A
  committed file symlink to a path outside the repository (`docs/notes.md -> ../../.ssh/id_ed25519`),
  or to an ignored path inside it (`notes.md -> .env`, `link -> .git`), was indexed and its
  contents could be returned by the MCP `search` tool. Symlinks are now followed only when
  their target is inside the repository and not ignored; `read_file` independently refuses
  to read outside the repository, protecting the snippet read at search time. Rows indexed
  through a bad symlink before this fix are purged automatically on the next sync, which
  `search` triggers. Repositories that deliberately symlink files in from outside the
  checkout will no longer have those files indexed.

## [0.2.0] - 2026-09-16

A ground-up rewrite. It keeps the purpose — let an agent find code when the user can't name
the symbol or module — and replaces everything else: no server process, no Docker, no
registry file that can drift from reality. Everything for a repository now lives in one local
SQLite file.

### Added

- **A local code graph**: `contains`, `imports`, `calls` and `inherits` edges, resolved from
  parsed references, alongside hybrid (semantic + keyword) search — answering not just "what
  is this about" but "what calls this" and "what breaks if I change it".
- **`neighbors` MCP tool**: walk the graph from a node ID returned by `search`, 1–3 hops, in
  either direction, along any combination of edge kinds.
- **Ten-language tree-sitter parsing**: Python, JavaScript, TypeScript, Rust, HTML, CSS, JSON,
  YAML, TOML and Markdown.
- **An agent skill** (`indexter skill --install`) teaching an agent when to reach for `search`
  and `neighbors` instead of grepping, and how to phrase a query.
- **`indexter --version`**.

### Changed

- **Storage**: one SQLite file per repository (vectors via sqlite-vec, keyword via FTS5, plus
  the graph) replaces Qdrant — no vector database to host, no container to run.
- **CLI**: reduced to setup only — `init`, `reindex`, `list`, `remove`, `mcp`, `skill`. Search
  happens through the MCP tools now, not the CLI.
- **MCP tools**: `search` and `neighbors` replace `list_repos`, `get_repo`, `search` and the
  `code_search_guide` prompt. stdio is the only transport.
- **Configuration**: a layered `~/.config/indexter/config.toml` plus a per-repository
  `indexter.toml` (or `[tool.indexter]` in `pyproject.toml`) replace the old XDG/env-var
  scheme; unknown keys are rejected with an error naming the key and file.
- **Default embedding backend** is now sentence-transformers, with `fastembed` available
  through the `onnx` extra (previously the reverse).

### Removed

- **Qdrant**, and its Docker-managed and in-memory storage modes.
- **The `repos.json` registry** — indexed repositories are discovered from their database
  files directly.
- **The streamable-HTTP MCP transport** and the `code_search_guide` prompt.
- **The `sync`, `search`, `status`, `settings`, `config-path` and `qdrant` CLI commands.**
- **The programmatic `Repo` class** and library API.
- **The modular install extras** (`[all]`, `[cli]`, `[mcp]`, `[core]`) — there is one package
  now, plus the optional `onnx` extra.

### Upgrading from 0.1

This release replaces the 0.1.x tool wholesale; nothing migrates automatically.

- Delete any 0.1 keys from `~/.config/indexter/config.toml` (for example `default_root`) —
  this version rejects unknown configuration keys, naming the key and file in the error.
- Stop and remove the Qdrant container and its data if you have no other use for it; this
  version never starts or talks to Qdrant.
- Re-index each repository with `indexter init <path>` — the database format and location
  are new, and nothing from a 0.1 install is read.
- Re-register the MCP server with your client — see the README for the current command and
  per-client configuration.

## [0.1.2] - 2026-01-19

### Changed

- **Release versioning**: Each release candidate now gets its own unique version, with the final release as a separate version
- **CI**: Disabled caching in publish jobs for more reliable builds

## [0.1.1] - 2026-01-19

### Added

- **Semantic parsing** via tree-sitter with support for 10 languages: Python, JavaScript, TypeScript, Rust, HTML, CSS, JSON, YAML, TOML, and Markdown
- **Hybrid search** combining dense semantic vectors (FastEmbed) with sparse BM25 vectors, fused via Reciprocal Rank Fusion (RRF)
- **Qdrant vector database integration** with Docker-managed and in-memory storage modes
- **Incremental indexing** with SHA-256 content hash change detection
- **Intelligent file walking** respecting `.gitignore`, with binary/minified file detection
- **CLI commands**: `init`, `sync`, `search`, `status`, `remove`, `settings`, `config-path`, `qdrant`
- **MCP server** (Model Context Protocol) for AI agent integration via FastMCP
  - Tools: `list_repos`, `get_repo`, `search`
  - Prompt: `code_search_guide`
  - Transports: stdio (default) and streamable-http
- **Multi-repository support** with separate Qdrant collections per repository
- **Modular installation**: `[all]`, `[cli]`, `[mcp]`, or `[core]` extras
- **XDG-compliant configuration** with global settings, per-repo overrides, and environment variable support
- **Programmatic API** via `Repo` class for library usage

### Notes

- Requires Python 3.11–3.13
- Requires Docker for Qdrant (default mode)
- Alpha release — API may change in future versions
