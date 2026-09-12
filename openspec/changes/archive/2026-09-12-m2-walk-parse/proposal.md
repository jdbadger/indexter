## Why

M1 built a database with the right tables and nothing to put in them. M2 produces the facts: a synchronous walker that decides which files count, and parsers that turn each file into `ParsedNode`s (the symbols) and `ParsedRef`s (the calls, imports, and inheritance that become graph edges in M4). This is the milestone where the graph either has the right raw material or doesn't — the resolution tiers in M4 can only be as good as the references extracted here.

The ten language parsers and the walker are lifted from `~/dev/indexter`, the only code carried over from v0.1.2. They work, but they were built to feed a flat vector store and carry three defects that a graph exposes: scope walks that return only the nearest enclosing class, so nested functions, JS callbacks, and Rust trait impls collide into one identity; a `Query` recompiled on every `parse()` call; and no reference extraction at all.

## What Changes

- Add `walk.py`: the lifted walker made **synchronous** (drops `anyio`), decoupled from the old `Repo` pydantic model — it takes a repository path and `Settings`. Keeps every filtering layer (gitignore patterns, binary extensions, minified detection, size limits, empty files, encoding fallback), the symlink-escape guard, and `sha256(relpath:content)` hashing. **BREAKING** relative to v0.1.2: `Walker.walk()` is an ordinary generator, not an async iterator.
- Add `parse/models.py`: `ParsedNode` and `ParsedRef`, replacing the `Document`/`NodeMetadata` pair. Fields line up with the M1 `nodes` and `refs` tables so M3's writer is a straight mapping.
- Add `parse/base.py`: `BaseParser`/`BaseLanguageParser` with **both** tree-sitter queries compiled once per parser instance (definitions and references), not per `parse()` call.
- Lift the ten language parsers (`python`, `javascript`, `typescript`, `rust`, `markdown`, `json`, `yaml`, `toml`, `html`, `css`) plus the `chunk.py` fallback, re-targeted to emit `ParsedNode`/`ParsedRef`.
- **Fix the three scope collisions** (all three reproduced against tree-sitter 0.26 before writing this):
  - Python: scope walk returns the full ancestor path, so `outer.inner` no longer collapses to `inner`.
  - JavaScript/TypeScript: same, plus object-literal methods and named callbacks that currently land at file scope.
  - Rust: include the `trait` field of `impl_item`, so `impl Display for Foo` and `impl Debug for Foo` stop producing two identical `Foo.fmt` identities.
- Add **reference extraction**: a second query per code parser capturing calls, imports, and inheritance, taking the head identifier of nested attribute chains (`self.x.y()` → head `self`; `os.path.join()` → head `os`).
- Add **deterministic node IDs**: `<relpath>::<scope path>.<name>#<kind>`, with `~N` suffixes for genuine duplicates ordered by line.
- Add a **normalized kind vocabulary**. The lifted parsers emit ad-hoc kinds (`"Header 1"`, `"h1"`, `"@media"`, `"mapping"`, `"table"`); these collapse to the settled set (`file`, `class`, `function`, `method`, `constant`, `interface`, `type_alias`, `enum`, `struct`, `trait`, `section`, `data`, `chunk`).
- **BREAKING** relative to v0.1.2: imports and exports stop being nodes. Python's `import` node type and TypeScript's `export` node type become `ParsedRef`s (imports) or attributes of the exported symbol (exports) — per settled decision 1, imports/exports are edges, not nodes.
- Every parsed file additionally yields one `file` node, which is what `contains` edges hang off in M4.
- Add `tests/fixtures/`: a small multi-language fixture repo containing the collision cases deliberately, plus per-language snapshot tests.

## Capabilities

### New Capabilities
- `file-walking`: Which files in a repository get indexed — the filtering layers, traversal and symlink guards, content hashing, and the synchronous iteration contract.
- `source-parsing`: Turning one file's contents into `ParsedNode`s — parser selection by extension, query compilation and caching, the normalized kind vocabulary, per-language extraction, and the chunk fallback.
- `node-identity`: Scope paths and the deterministic node ID format, including duplicate disambiguation and the collision cases that motivate it.
- `reference-extraction`: Turning one file's contents into `ParsedRef`s — calls, imports, and inheritance, with head-identifier resolution for attribute chains.

### Modified Capabilities
<!-- None. M2 writes nothing to the database, so the M1 schema, config, paths, and CLI specs are unchanged. -->

## Impact

- **New code**: `src/indexter/walk.py`, `src/indexter/parse/` (`__init__.py`, `models.py`, `base.py`, ten language modules, `chunk.py`, `ids.py`), with co-located tests and `tests/fixtures/`.
- **Configuration**: `Settings` gains the walker's knobs — `max_file_size_bytes` and `ignore_patterns` already exist from M1; the chunk fallback's size and overlap are added.
- **Not touched**: the database. M2 produces in-memory `ParsedNode`/`ParsedRef` values and writes nothing — M3 owns the writer, so `db/` is unchanged and no schema version bump is needed.
- **Downstream**: M3 composes and embeds these nodes; M4 resolves these refs into edges. Anything M2 fails to extract is invisible to both, which is why reference extraction is specified here in detail rather than left to M4.
- **Deferred to M4 by design**: the builtin/stdlib stoplist. Settled decision 9 drops those references "at extraction", but the milestone table places the stoplist in M4 — M2 therefore extracts every reference and leaves a documented filter seam for M4 to fill.
- **Dependencies**: no new ones. `tree-sitter`, `tree-sitter-language-pack`, and `pathspec` are already declared; `anyio` is deliberately not adopted.
