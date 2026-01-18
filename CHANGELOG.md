# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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