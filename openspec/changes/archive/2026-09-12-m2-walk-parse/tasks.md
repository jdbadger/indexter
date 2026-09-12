## 1. Models and scaffolding

- [x] 1.1 Create the `src/indexter/parse/` package with `__init__.py` and co-located `tests/`
- [x] 1.2 Define the `Kind` enum in `parse/models.py`: `file`, `class`, `function`, `method`, `constant`, `interface`, `type_alias`, `enum`, `struct`, `trait`, `section`, `data`, `chunk` (plus `external_module` reserved for M4, documented as never emitted by parsing)
- [x] 1.3 Define the `RefKind` enum: `calls`, `imports`, `inherits`
- [x] 1.4 Define `ParsedNode` as a frozen slotted dataclass — `kind`, `name`, `scope_path: tuple[str, ...]`, `language`, `start_line`, `end_line`, `start_byte`, `end_byte`, `signature`, `docstring`, plus `id` and `parent_id` assigned later; no body field
- [x] 1.5 Define `ParsedRef` as a frozen slotted dataclass — `from_node_id`, `raw_name`, `head`, `ref_kind`, `line`, `col`; no resolved target or confidence
- [x] 1.6 Define `ParseResult` carrying `nodes`, `refs`, and `errors: list[str]`
- [x] 1.7 Add `Settings` keys for the chunk fallback (`chunk_size`, `chunk_overlap`) with the lifted defaults
- [x] 1.8 Tests: enum membership is closed, dataclasses are frozen and reject mutation, `ParseResult` defaults to empty collections

## 2. Walker (`walk.py`)

- [x] 2.1 Lift `IgnorePatternMatcher` from the old walker unchanged in behaviour (pattern add, add-from-file, `should_ignore`)
- [x] 2.2 Lift `BINARY_EXTENSIONS` and the minified-name check
- [x] 2.3 Define `WalkedFile` (relative path, size, mtime, extension) — no content
- [x] 2.4 Implement `Walker(repo_path, settings)`, building the matcher from configured patterns plus the repo `.gitignore`; drop the old `Repo` model coupling
- [x] 2.5 Implement synchronous `walk()` as a generator: recursive traversal, directory pruning for ignored dirs, symlink-escape guard, tolerance for permission errors and vanished files — drop `anyio` entirely
- [x] 2.6 Apply the content-independent filters during the walk: ignore patterns, binary extensions, minified names, max size, zero-byte
- [x] 2.7 Implement `read_file(repo_path, relpath)` returning decoded text plus `sha256(relpath:content)`, with UTF-8 then Latin-1 fallback, returning `None` when undecodable
- [x] 2.8 Build the walker test fixture programmatically in `tmp_path`: nested dirs, a `.gitignore`, an ignored dir, a binary file, a minified file, an oversized file, an empty file, an undecodable file, an internal symlink, an escaping symlink, a broken symlink
- [x] 2.9 Tests — traversal: relative forward-slash paths, lazy generator with no event loop, stat fields populated, empty repository, no contents read during a walk
- [x] 2.10 Tests — filtering: configured patterns, `.gitignore` patterns, ignored directory pruned rather than descended, missing `.gitignore` is fine, binary/minified/oversized/empty all skipped
- [x] 2.11 Tests — safety: escaping symlink not followed, internal symlink fine, broken symlink skipped, unreadable directory skipped without raising, file vanishing between listing and stat
- [x] 2.12 Tests — reading: content and hash returned, hash changes with content, hash changes with path, Latin-1 fallback, undecodable returns `None`

## 3. Node identity (`parse/ids.py`)

- [x] 3.1 Implement `build_id(relpath, scope_path, name, kind)` producing `<relpath>::<scope>.<name>#<kind>`, omitting the scope segment at file scope
- [x] 3.2 Implement the file-wide duplicate pass: group nodes by identical ID, sort by start line, append `~2`, `~3`, … leaving the first unsuffixed
- [x] 3.3 Implement the parent-linking pass: each node's parent is the innermost enclosing node in the same file by scope path; file-scope nodes parent to the `file` node; the `file` node has none
- [x] 3.4 Wire both passes into the file-level parse entry point so parsers never assign IDs themselves (done as part of 4.7's `parse_file()`)
- [x] 3.5 Tests — format: method inside class, function at file scope, file node, reproducibility across two parses
- [x] 3.6 Tests — stability: inserting lines above a symbol leaves the ID unchanged, editing a body or docstring leaves it unchanged, renaming changes it, moving into a class changes it
- [x] 3.7 Tests — duplicates: two same-named functions get `~2` on the later one, numbering follows line order not emission order, different kinds are not duplicates, unique names never suffixed
- [x] 3.8 Tests — parent linking: method to class, file-scope symbol to file node, file node has no parent, deeply nested function links to its immediate parent

## 4. Parser framework (`parse/base.py`)

- [x] 4.1 Implement `BaseParser` with a `parse(relpath, content) -> ParseResult` interface, replacing the old `Document`-coupled signature
- [x] 4.2 Implement `BaseLanguageParser`: load the tree-sitter language and parser once, compile the definitions query and the optional references query in `__init__`, create a fresh `QueryCursor` per parse
- [x] 4.3 Implement the shared scope-walk helper returning the full ancestor path, parameterized by each language's scope-forming node types and how each names them
- [x] 4.4 Implement the shared head-identifier helper: walk the leftmost spine of an attribute/member chain to its base identifier, returning `None` when the base is a call, literal, or subscript
- [x] 4.5 Implement the `file`-node emission shared by every parser
- [x] 4.6 Implement per-file error containment: catch unexpected parser exceptions, record them in `ParseResult.errors`, return what was produced
- [x] 4.7 Implement the extension registry and `parse_file()` entry point, caching one parser instance per language and falling back to the chunk parser
- [x] 4.8 Tests — framework: queries compiled once across several parses, repeated parses independent, parser instances reused, case-insensitive extension match, unregistered extension falls back
- [x] 4.9 Tests — containment: syntax-error file still yields parsed constructs and reports the problem, a parser raising is contained, empty file yields cleanly

## 5. Fixture repository

- [x] 5.1 Create `src/indexter/parse/tests/fixtures/` as a committed multi-language tree
- [x] 5.2 Python fixtures: nested functions (`outer.inner`), two `inner`s under different parents, a duplicate same-name function, a decorated function, a documented class with a base, imports (plain, from, relative), `self.x.y()` and `os.path.join()` calls, a module constant
- [x] 5.3 JavaScript fixtures: class with method, named callback inside a method, two object literals each with a `handler` method, arrow functions, imports, `extends`
- [x] 5.4 TypeScript fixtures: exported declarations, interface, type alias, enum, class with `extends` and `implements`
- [x] 5.5 Rust fixtures: `impl Display for Foo` and `impl Debug for Foo` both defining `fmt`, an inherent `impl`, a trait, a struct, `use` declarations
- [x] 5.6 Non-code fixtures: Markdown with nested headings, JSON, YAML, TOML, HTML, CSS, and one file with an unregistered extension for the chunk fallback
- [x] 5.7 Add ruff `extend-exclude` and pytest collection ignore for the fixture directory, so deliberately odd fixture code is neither linted nor collected
- [x] 5.8 Confirm the fixture tree stays out of the wheel (already covered by `wheel-exclude`) and that no fixture file shadows a real module name

## 6. Code parsers: lift and fix

- [x] 6.1 Lift `python.py`: keep the definitions query and decorated-definition suppression, emit `ParsedNode`s with normalized kinds, drop the `import` node type
- [x] 6.2 Fix the Python scope walk to return the full ancestor path
- [x] 6.3 Add the Python references query: calls (bare and attribute chains), `import`/`from ... import` including relative-import dots, class bases
- [x] 6.4 Lift `javascript.py`, normalize kinds, drop export nodes
- [x] 6.5 Fix the JavaScript scope walk: full ancestor path, named callbacks inside methods, object-literal methods scoped to their binding name
- [x] 6.6 Add the JavaScript references query: calls, `import`/`require` specifiers, `extends`
- [x] 6.7 Lift `typescript.py`, keeping export-wrapped duplicate suppression, emitting `interface`, `type_alias`, and `enum` kinds
- [x] 6.8 Fix the TypeScript scope walk to match JavaScript's, and add its references query including `implements`
- [x] 6.9 Lift `rust.py`, normalize kinds (`struct`, `trait`, `enum`, `function`, `method`, `constant`)
- [x] 6.10 Fix the Rust scope segment to include the trait: `Foo<Display>` from `impl std::fmt::Display for Foo`, using the trait path's last segment; plain `impl Foo` stays `Foo`
- [x] 6.11 Add the Rust references query: calls, `use` declarations, `impl ... for ...` as inheritance
- [x] 6.12 Tests — collisions (hand-written, not snapshots): Python `outer.inner` distinct from file-scope `inner`; two `inner`s under different parents distinct; JS `cb` scoped `A.m`; two object-literal `handler`s distinct; Rust `Foo<Display>.fmt` distinct from `Foo<Debug>.fmt`
- [x] 6.13 Tests — references (hand-written): `self.validate` head `self`; `os.path.join` head `os`; bare call head equals raw name; `build().run()` recorded with no head; relative import dots preserved; multiple bases yield one ref each; every ref's origin ID matches a node from the same parse
- [x] 6.14 Tests — nodes: decorated Python function yields one node whose range includes the decorator; exported TS declaration yields one node; no import or export nodes anywhere; byte ranges slice back to the exact source text

## 7. Non-code parsers: lift and normalize

- [x] 7.1 Lift `markdown.py`, mapping heading levels to kind `section` and carrying the heading path in the name
- [x] 7.2 Lift `json.py`, `yaml.py`, `toml.py`, mapping their blocks to kind `data`
- [x] 7.3 Lift `html.py` and `css.py`, mapping elements and rules to kind `section`
- [x] 7.4 Lift `chunk.py` as the fallback, emitting kind `chunk`, driven by the new settings keys
- [x] 7.5 Confirm none of these parsers emit references
- [x] 7.6 Tests: every node from every non-code fixture has a kind in the closed vocabulary, no references are produced, chunk fallback covers the whole file

## 8. Snapshot suite

- [x] 8.1 Write the snapshot harness: parse one fixture file, render nodes and refs deterministically (sorted, with volatile fields excluded)
- [x] 8.2 Snapshot each Python, JavaScript, TypeScript, and Rust fixture
- [x] 8.3 Snapshot each Markdown, JSON, YAML, TOML, HTML, CSS, and fallback fixture
- [x] 8.4 Add a reference-count assertion over the Python fixtures, so a regression that silently stops extracting refs fails loudly rather than just re-snapshotting

## 9. Verification

- [x] 9.1 `uv run --group dev ruff check --fix src/indexter` clean
- [x] 9.2 `uv run --group dev ty check src/indexter` clean
- [x] 9.3 `uv run --group test pytest --cov=indexter --cov-fail-under=95 --cov-report=term-missing` green
- [x] 9.4 Real-repo smoke check: walk `~/dev/indexter`, confirm the file count is sane and that no file is read during the walk
- [x] 9.5 Real-repo smoke check: parse every walked file, confirm no unhandled exceptions, report per-kind node counts and per-kind reference counts
- [x] 9.6 Compare the Python reference count against the plan's measured baseline (~1,079 call refs over 34 files) and record the actual figure for M4 to resolve against
- [x] 9.7 Confirm the suite passes on Python 3.11, 3.12, and 3.13
