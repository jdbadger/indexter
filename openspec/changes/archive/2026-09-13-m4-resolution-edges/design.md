## Context

After M3 a synced database holds nodes with stable text IDs and parent links, and every call, import and inheritance reference in `refs` with status `unresolved`. `edges` is empty and `nodes.degree` is 0 everywhere. M4 turns references into edges, which is where settled decision 8 ("`refs` are the durable source of truth and edges are derived from them") and decision 9 (resolution tiers, 5-candidate cap, builtin stoplist) become code.

Measured on this machine before writing this, against the M3 index of `~/dev/indexter` (73 files, 2,711 nodes, 7,815 refs — all Python) with a throwaway prototype of the tiers below:

| Measurement | Result |
|---|---|
| Refs by kind | 7,328 calls · 426 imports · 61 inherits |
| Imports resolving to a repo module / external / module found but member not | 182 / 214 / 30 |
| Calls whose head is a Python builtin | 1,218 (16.6%) |
| Calls whose head is bound by an import to an external package (`Mock`, `patch`, `Path`) | 2,392 (32.6%) |
| High-confidence (exact / imported / unique_name, external counted as imported) of non-builtin calls | 72.5% overall · 92% of `src` call **edges** |
| Ambiguous (2–5 candidates) of non-builtin calls | 5.9% |
| Most common unresolved `src` calls | `console.print` (87), `logger.debug/info/warning` (53), `match.get`, `node.child_by_field_name` — methods on variables of external or builtin types |
| Most common ambiguous calls | `rust_parser.parse`, `html_parser.parse`, … — pytest fixture parameters whose tail name matches 5 parser methods |
| Without narrowing candidates to classes the file imports, share of call edges at high confidence | 46% (ambiguous fan-out dominates) |
| Full in-memory resolution of all 7,815 refs | ~65 ms |
| Rewrite 3,000 edges + update every ref + recompute degree, one transaction | ~37 ms |

Two gaps in M2's extraction surfaced while prototyping, and M4 has to close them:

1. **Import refs don't say what name they bind.** `from a import b` and `import a.b` both produce `raw_name = "a.b"` with no head; aliases (`import numpy as np`, `from x import y as z`) are lost; JavaScript records one ref per statement with only the specifier, so `import {foo as bar} from './m'` can never resolve `bar()`.
2. **Rust `impl Display for Foo` originates from the file node**, because the `impl` block isn't a node, and nothing records `Foo` — the `inherits` edge the spec promises from `Foo` can't be built.

Constraints: synchronous; sync-on-search (decision 5) means the no-op path must stay free and the one-file-changed path well under a second; the database is a rebuildable cache (M1 decision 4); tests co-located, ≥95% coverage.

## Goals / Non-Goals

**Goals:**
- Every ref gets a deterministic outcome with a confidence, and every outcome that names targets yields edges consistent with it.
- `contains`, `calls`, `imports` and `inherits` edges, with line and confidence, plus `nodes.degree`.
- Per-language module resolution good enough that imports between repository files resolve for Python, JavaScript/TypeScript and Rust layouts in common use.
- `external::<name>` nodes so "what touches pydantic?" is one hop.
- After any completed sync, no edge and no resolved ref names a missing node, and failed refs have been retried.
- Resolution cost paid only by syncs that changed something.
- Real-repo resolution near the plan's ~80% high-confidence rate, with a report precise enough to see where the misses are.

**Non-Goals:**
- Type inference: receivers assigned from constructors (`w = Walker(p); w.walk()`), parameter annotations, return types. The largest remaining `src` miss (`console.print`, `logger.debug`) needs it; see Open Questions.
- `references`, `type_of`, `instantiates` edges (settled decision 1).
- Resolving across repositories, or into installed packages' source.
- TypeScript `paths` aliases, `package.json` `exports`/workspaces, Cargo workspaces resolving sibling crates by name — treated as external (Risks).
- Search-time use of edges: expansion, hub damping, roll-up (M5).
- Macro-generated calls, dynamic dispatch, `getattr`, decorators' runtime effects.

## Decisions

### 1. Resolve the whole repository whenever anything changed, and write only the difference

Resolution loads every node and ref into memory, computes every ref's outcome and the complete desired edge set, compares with what is stored, and writes the changes in one transaction: changed ref outcomes, inserted and deleted edges, changed edge confidences, external nodes added or removed, and degrees of nodes whose incident edges changed.

*Why whole-repo:* outcomes are not local to the changed file. Tier 4 depends on how many nodes in the repository share a name — adding `def parse` anywhere turns another file's `unique_name` into `ambiguous`. Narrowing depends on which classes a file imports, and imports resolve against other files' contents. Re-export chains cross files. Scoping resolution by the sync report would need an invalidation index for each of these, and a missed case is a silently wrong edge. The prototype resolves this repository in ~65 ms; the plan's largest local repository is about the same size.

*Why diff-writes:* a one-line edit changes a handful of outcomes. Rewriting 7,815 ref rows and all edges on every edit churns the WAL for nothing, and unchanged rows keep their `edges.id`.

*Consequence:* M3 decision 10 said the sync report would scope M4. It is used as a gate (resolve only when something changed) rather than as a scope.

*Alternative rejected:* per-file resolution scoped by changed names. Faster on very large repositories, and the right later optimization if one shows up (Risks), but its correctness depends on enumerating every cross-file dependency of an outcome.

### 2. Resolution is a separate step inside sync, gated by a pending marker

`sync_repo` becomes: pass one (structural writes, M3) → resolution → embedding backlog. Resolution runs when either:

- `project_metadata.resolution_pending` is set — written inside the same transaction as every structural write (`write_file`, `remove_file`, `record_unreadable`) and cleared inside resolution's own transaction; or
- `project_metadata.resolver_version` differs from the code's `RESOLVER_VERSION`.

*Why a marker instead of "the report says something changed":* a process killed after pass one's commits but before resolution leaves unresolved refs and dangling edges, and the next sync's pass one sees nothing changed. The marker makes "resolution owed" a fact in the database, the same way M3 decision 1 made "embedding owed" a fact (nodes without vectors). A touched-but-identical file writes no structural rows and doesn't set it.

*Why a resolver version:* improving the resolver shouldn't require re-parsing, so it is not part of M3's index fingerprint. Bumping `RESOLVER_VERSION` re-resolves on the next sync with no parse and no embedding.

*Why before the backlog:* the backlog is the slow part when it has work (model load), and structure should be complete as early as possible. External nodes are excluded from the backlog (decision 9), so ordering doesn't affect embedding.

### 3. Import references carry a binding: `head` is the bound name, `imported_name` the member

Import refs are emitted **one per bound name**:

| Source | `raw_name` | `imported_name` | `head` |
|---|---|---|---|
| `import a.b.c` | `a.b.c` | — | `a` |
| `import a.b as x` | `a.b` | — | `x` |
| `from a.b import c as d` | `a.b` | `c` | `d` |
| `from . import sibling` | `.` | `sibling` | `sibling` |
| `from a import *` | `a` | `*` | — |
| `import X, {a as b} from './m'` | `./m` | `default` / `a` | `X` / `b` |
| `import * as ns from 'react'` | `react` | — | `ns` |
| `import './side-effect'` | `./side-effect` | — | — |
| `const q = require('./r')` | `./r` | — | `q` |
| `export {d as e} from './y'` | `./y` | `d` | `e` |
| `export * from './z'` | `./z` | `*` | — |
| `use crate::auth::Handler` | `crate::auth` | `Handler` | `Handler` |
| `use std::io as sio` | `std` | `io` | `sio` |
| `use serde` | `serde` | — | `serde` |
| `use a::b::*` | `a::b` | `*` | — |

For calls and inheritance `head` keeps its M2 meaning: the leftmost identifier of the chain. For imports it becomes the name the import introduces into scope. Both are "the in-file name resolution goes through", which is exactly what tier 3 joins on: a call whose head is `d` looks for an import ref whose head is `d`.

*Why a column, not an encoding in `raw_name`:* the prototype had to guess plain-import vs from-import from `a.b` and got `from datetime import datetime` wrong. JavaScript needs specifier and member together. The schema is a cache (decision 10 covers the bump).

*Why Python splits module and member, and Rust splits the last segment:* Python's syntax says which is the module; Rust's doesn't (`crate::auth::Handler` may be a module or an item), so Rust uses the same split and resolution tries "module path + member as a submodule" before "member inside module path" — the same order Python uses, since `from pkg import sub` may also name a submodule.

*Re-exports* (`export … from`) are recorded as imports of the re-exporting file, which is true, and let decision 6 follow barrels the same way it follows Python `__init__.py` imports.

### 4. Rust trait impls record the implementing type in `for_type`

`impl Display for Foo` yields an `inherits` ref with `raw_name = Display`, `head = Display`, and `for_type = Foo`, still originating from the enclosing node (the file). Resolution resolves `for_type` with the same name tiers used for any head (same file, imports, unique name among `struct`/`enum`/`type_alias`/`trait`) and uses that node as the edge source.

*Why not originate from `Foo`:* extraction only sees one file, and `Foo` may be defined in another; M2's invariant is that every ref's origin is a node from the same parse.

The same `for_type` resolution gives tier 1 its class for Rust: a method in scope `Foo<Display>` resolves `self`/`Self` to the type named by the scope segment with the `<Trait>` suffix removed.

Rust methods are not children of their type's node — their parent is the file, and `impl` blocks may sit in other files than the `struct`. So wherever the tiers below look up "a type's members", a Rust type's members are the methods, in any file of the same language family, whose scope path is the type's name with or without a `<Trait>` suffix.

### 5. The tiers

Each ref is resolved by the first tier that produces an outcome. The ref's language family (`python`, `javascript`+`typescript`, `rust`) restricts every candidate set; references never resolve across families.

**Tier 1 — receiver is `self`/`cls`/`this`/`Self`** (`exact`). The enclosing class is found by walking the origin's parents (for Rust, decision 4's type). The chain's next segment is looked up among that class's members, then among its bases' members, following resolved `inherits` edges breadth-first with cycle protection. Inherited members are still `exact`. A chain longer than one member (`self.store.add`) or a miss falls through to tier 4.

**Tier 2 — head defined in an enclosing scope** (`exact`). Walk from the origin through its ancestors to the file node, looking for a child named `head`. Class-like containers (`class`, `struct`, `trait`, `interface`, `enum`) are skipped unless the origin is the container itself — a method body can't name a sibling method bare in any of the four languages. If the chain has more segments, look each one up among the found node's children (class members, including inherited, for classes). A head that names a function or constant while the chain continues (`fixture_fn.parse()` — a pytest parameter shadowing a same-named fixture function) falls through to tier 4 rather than resolving to nonsense.

**Tier 3 — head bound by an import, or a module-qualified path** (`imported`). Look for an import binding with that `head` whose origin is the ref's origin or an ancestor of it (innermost wins, so function-local imports shadow file-level ones). The binding's target (decision 6) is then walked by the chain's remaining segments: into a module's top-level nodes or submodules, into a class's members. A target outside the repository makes the ref `external` with `resolved_target_id = external::<name>`. Unbound heads are also tried against wildcard imports' modules, and — for Rust — as a path from the current module (`auth::login()` after `mod auth;`) or from `crate`/`self`/`super`.

**Tier 4 — unique name** (`unique_name`) and **tier 5 — ambiguous**. The chain's last segment is looked up among candidate nodes of the kinds the ref can target (`calls`: function, method, class, struct; `inherits`: class, interface, trait). Candidates are first **narrowed** to members of the classes the file defines or imports (and their bases); only if that set is empty is the lookup repo-wide. One candidate → `resolved`/`unique_name`; 2–5 → `ambiguous`, candidate IDs in `candidates`; more than 5 → `too_ambiguous`, candidate IDs in `candidates` (first 20 by ID), no edges; none → `failed`.

**Tail stoplist.** The repo-wide lookup skips attribute calls on an unresolved receiver whose last segment is a method of the language's builtin types (Python `get`/`append`/`items`/`join`, JavaScript `push`/`map`/`then`, Rust `unwrap`/`clone`/`iter`, …) — they become `failed`. Narrowed lookups don't apply it: `store.add()` in a file that imports `VectorStore` should still find `VectorStore.add`.

*Why narrowing:* measured. Without it, fixture-parameter calls like `rust_parser.parse()` fan out to every `parse` method and only 46% of call edges are high-confidence; with it, 73%. A file that imports `RustParser` calling `.parse()` on something is overwhelmingly calling `RustParser.parse`.

*Why narrowed-unique is `unique_name`, not `imported`:* it is still an inference about an untyped receiver, just over a smaller population. `imported` is reserved for refs whose head is actually bound.

*Why the stoplist only on repo-wide tails:* bare builtins are already dropped at extraction (decision 8); the tail stoplist exists because `match.get(...)` would otherwise land on whichever repository class happens to define `get`.

### 6. Module resolution per language

A binding's target is resolved once per resolution run and memoized.

**Python.** Relative modules (`.`, `..pkg`) resolve against the importing file's package directory. Absolute dotted paths are matched **by suffix** against repository files — `indexter.config` matches `src/indexter/config.py` or `src/indexter/config/__init__.py` at a path-component boundary — so `src/` layouts, flat layouts and monorepo subpackages need no configuration. Several matches prefer the longest common directory prefix with the importer, then the shortest path. For `from m import n`: `m.n` as a module first, then a top-level node `n` in `m`, then `n` among `m`'s own import bindings (a re-export in `__init__.py`), following at most 5 hops; if `m` resolves but `n` doesn't, the target is `m`'s file node (the name exists at runtime, perhaps as a lowercase variable, but isn't a node). An absolute path with no match is external, named by its first segment.

**JavaScript/TypeScript.** Specifiers starting `./` or `../` resolve against the importer's directory: the exact path, then with `.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, `.cjs` appended, then `index` with each; a specifier ending `.js`/`.jsx`/`.mjs`/`.cjs` is also tried with the TypeScript counterpart (TypeScript's ESM convention). Anything else is external, named by the package: `@scope/name` or the first path segment, `node:` prefix kept. A `default` member resolves to a top-level node named like the binding, else the module's only top-level class or function, else the module's file node.

**Rust.** A file's module path comes from its position under its crate root — the nearest ancestor directory holding `lib.rs` or `main.rs` — with `mod.rs`, `lib.rs` and `main.rs` naming their directory. `crate::a::b` resolves to `a/b.rs` or `a/b/mod.rs` under the root, `self::` against the current module, `super::` against its parent, and a path starting with any other name first as a child module of the current module and otherwise as an external crate named by that segment (`std`, `serde`). The last segment is tried as a submodule before as an item, as in Python.

*Why suffix matching for Python:* configuring source roots is the kind of setup the rewrite is removing; the ambiguity it introduces (two files with the same dotted suffix) is rare and resolved deterministically.

### 7. Outcomes, statuses and the edges they produce

| Status | `confidence` | `resolved_target_id` | `candidates` | Edges |
|---|---|---|---|---|
| `resolved` | `exact` / `imported` / `unique_name` | target | — | one, at that confidence |
| `external` | `imported` | `external::<name>` | — | `imports` refs only |
| `ambiguous` | `ambiguous` | — | 2–5 IDs | one per candidate, `ambiguous` |
| `too_ambiguous` | — | — | up to 20 IDs | none |
| `failed` | — | — | — | none |

Edge source is the ref's origin (decision 4's type for Rust impls); line is the ref's line. An `imports` ref's target is the most specific thing resolved — the imported symbol, else the imported submodule's file node, else the module's file node, else the external node.

*Why calls into external packages get no edge:* the file's `imports` edge to `external::unittest` already answers "what touches it" at one hop; 2,392 `calls` edges from test methods into one `external::unittest` node would make it the largest hub in the graph and tell search nothing the import doesn't. The ref still records the external target, so the fact isn't lost.

`contains` edges are derived from `nodes.parent_id` for every node, with no line and confidence `exact`.

### 8. Builtins are dropped at extraction, unless the file shadows them

A call or inheritance ref whose head is in the language's builtin list is not emitted — Python's `builtins` names (frozen as a literal, not read from the running interpreter), JavaScript/TypeScript globals (`console`, `JSON`, `Math`, `Object`, `Array`, `Promise`, `Error`, `setTimeout`, …), Rust prelude names (`Some`, `None`, `Ok`, `Err`, `Box`, `Vec`, `String`, `Option`, `Result`, `drop`). The exception: a head the same file defines as a node or binds by an import is kept, since then it isn't the builtin.

*Why at extraction:* the plan says so, and it is right — 1,218 of 7,328 call refs here are builtins; storing, resolving and reporting them as failures every sync buys nothing. Extraction already has the file's nodes and import bindings in hand, which is exactly the shadowing check.

*Why frozen lists:* the parse output must not depend on the interpreter version running the indexer, or two machines would build different indexes from the same source. The lists are part of parse output, so changing them bumps `INDEX_FORMAT_VERSION`.

### 9. External modules are nodes with no file and no vector

`external::<name>` nodes have kind `external_module`, `name` and `qualified_name` equal to the package name, empty `file_path`, and no language (the same package name imported from Python and JavaScript is one node). Resolution inserts them, with a full-text row holding their name, when a ref first targets them, and deletes them (and their full-text row) when no ref targets them any more. The embedding backlog skips them: there is no text to embed beyond a name that full-text search already matches exactly.

### 10. Schema version 2: two nullable columns on `refs`

`refs` gains `imported_name TEXT` and `for_type TEXT`. `SCHEMA_VERSION` becomes 2 and `INDEX_FORMAT_VERSION` becomes 2. `init`/`reindex` already rebuild a database with a mismatched schema version (M3 decision 12), so existing databases are rebuilt on their next index — nothing to migrate.

### 11. Degree counts graph edges, not containment

`nodes.degree` is the number of `calls`, `imports` and `inherits` edges incident to the node, in either direction, at any confidence. `contains` is excluded: a large file or class isn't a hub because it contains a lot, and M5's hub damping ("skip expansion targets with degree > 40") is about fan-in and fan-out through the graph. Degrees are recomputed only for nodes whose incident edges changed in the diff.

### 12. Resolution reports what it did, and a query reports what the graph holds

`resolve_repo` returns a `ResolveReport`: ref counts by kind × status and confidence, edge counts by kind × confidence (after the run, not just the diff), external node count, edges inserted and deleted, and elapsed time. `SyncReport` gains `resolution: ResolveReport | None` (`None` when resolution didn't run). `init`/`reindex` print one extra summary line from it. `db/queries.py` gains `resolution_summary(conn)` computing the same counts from stored rows, for verification and for later milestones.

The real-repo acceptance numbers are **per ref**, the way the plan's baseline was measured: of recorded (non-builtin) `calls` refs, the share `resolved` at `exact`/`imported`/`unique_name` plus `external`, and the share `ambiguous`. Edge shares are reported alongside because M5 consumes edges.

### 13. A fixture repository asserts the graph

`src/indexter/index/tests/fixtures/graph_repo/` is a small multi-language tree with known answers: a Python package with absolute, relative, aliased and wildcard imports, an `__init__.py` re-export, nested functions, inherited `self` calls, a builtin shadowed by a local definition, and two same-named functions in different modules; a JavaScript/TypeScript tree with default, named, namespace and `require` imports, `index` and `.js`→`.ts` resolution, a barrel re-export, callbacks, `this` calls, `extends` and `implements`; a Rust crate with `crate::`/`self::`/`super::`, `mod.rs`, `impl Display for Foo` and `impl Debug for Foo`, and `Self::new()`. Tests sync it with `FakeEmbedder` and assert specific edges per tier, plus a rendered snapshot of every edge and every ref outcome, so a regression in any tier shows up as a diff.

## Risks / Trade-offs

- **Whole-repo resolution on every changed sync grows with repository size** → ~65 ms at 7.8k refs; a 100k-ref repository would pay ~1 s per edited file. Out of the plan's scale ("largest local repo: 75 source files"); the escape hatch is scoping by changed names (decision 1's rejected alternative), and `ResolveReport.elapsed_seconds` makes the cost visible.
- **`unique_name` on untyped receivers is a guess** → labeled as such, never `exact`; M5 can weight it. Narrowing and the tail stoplist remove the worst cases measured.
- **Suffix-matched Python modules can pick the wrong file** when two directories hold the same dotted path (vendored copies, `tests/fixtures` mirroring a package) → deterministic tie-break toward the importer's own tree; wrong only when the importer is equidistant.
- **TypeScript path aliases, `package.json` exports and workspace crates resolve as external** → imports of first-party code look external in those repositories. Visible in the report as a high external share; a later change can read `tsconfig.json`/`Cargo.toml` without schema changes.
- **Rust inline `mod x { … }` blocks aren't nodes** → paths into them fall through to name tiers.
- **Dynamic receivers stay `failed`** (`console.print`, `logger.debug`) → they are genuinely unresolvable without type inference, and are retried at no extra cost since resolution is whole-repo.
- **Between pass one and resolution, edges can dangle** → only inside a running sync or after a crash; search syncs in-process first (decision 5), and the pending marker repairs a crash on the next sync.
- **Parser changes invalidate every existing index** → schema bump rebuild; about 12 s for `~/dev/indexter`, dominated by the model.
- **The per-ref acceptance metric may land below 75% on test-heavy repositories** → the prototype measured 72.5%, with most misses in categories this design names as out of scope. The verification task records the actual breakdown rather than tuning to the number.

## Migration Plan

Existing M3 databases hold schema version 1. The first `indexter init` or `reindex` after this change detects the mismatch, deletes the database, and indexes from scratch (M3's existing path), producing edges on that same run. An MCP server doesn't exist yet, so nothing else opens databases. Rollback is checking out the previous version and running `reindex`, which rebuilds at version 1 the same way.

## Open Questions

- **Receiver type inference.** Local assignments from constructors and parameter annotations would resolve `walker.walk()`, `console.print()` (to `external::rich`) and most remaining `src` misses. It needs new extracted facts (assignments, annotations) and is left for after M5's eval shows whether missing call edges hurt retrieval.
- **Should `ambiguous` edges for methods collapse to their shared base method** (five `parse` implementations of one `BaseLanguageParser.parse`) instead of fanning out? Possibly better for search; decided after M5 sees the graph.
- **Cross-family resolution** (a TypeScript file importing a `.json` or `.css` module, Python loading a sibling `.sql`) is recorded as a failed or external import today. Whether those deserve `imports` edges to data/section nodes is left to the eval.
