# reference-resolution Specification

## Purpose

M2's extraction leaves every reference unresolved: it knows a call's raw text and head
identifier but not which node, if any, it names. Turning "calls `self.validate`" into an edge to
a specific method needs scope walking, import-binding lookups, and per-language module
resolution — none of which extraction, working one file at a time, can do.

This capability owns turning a reference into an outcome: the ordered tiers (`self`/class
members, definitions in enclosing scope, import bindings and module-qualified paths, then name
lookup narrowed to the classes a file uses and finally repo-wide), per-language module resolution
for Python, JavaScript/TypeScript, and Rust, the ambiguity cap and builtin-method stoplist that
keep name lookup from guessing wildly, and the `resolved`/`external`/`ambiguous`/`too_ambiguous`/
`failed` statuses and confidences that `graph-edges` turns into edges.

## Requirements

### Requirement: Every reference receives a resolution outcome

Resolution SHALL assign every stored reference exactly one status — `resolved`, `external`, `ambiguous`, `too_ambiguous`, or `failed` — replacing `unresolved`. A `resolved` reference SHALL carry one target node ID and a confidence of `exact`, `imported`, or `unique_name`. An `external` reference SHALL carry the external module node's ID as its target and confidence `imported`. An `ambiguous` reference SHALL carry no single target, confidence `ambiguous`, and its 2 to 5 candidate node IDs. A `too_ambiguous` reference SHALL carry no target and no confidence, and at most 20 of its candidate node IDs. A `failed` reference SHALL carry no target, confidence, or candidates. Outcomes SHALL be deterministic for the same repository contents.

#### Scenario: No reference is left unresolved

- **WHEN** a repository is synced
- **THEN** no stored reference has status `unresolved`

#### Scenario: Resolution is deterministic

- **WHEN** the same repository contents are indexed into two fresh databases
- **THEN** every reference has the same status, target, confidence, and candidates in both

#### Scenario: Candidate list is capped for overly ambiguous names

- **WHEN** a reference's name matches more than 20 candidate nodes
- **THEN** its status is `too_ambiguous` and it records exactly 20 candidate IDs

### Requirement: References resolve through ordered tiers

A reference SHALL be resolved by the first applicable tier, in order: (1) a `self`, `cls`, `this`, or `Self` receiver resolves to a member of the enclosing type or of its resolved base types, with confidence `exact`; (2) a head defined in an enclosing scope resolves to that definition, with its remaining chain segments looked up among the definition's members, with confidence `exact`; (3) a head bound by an import visible from the reference's origin, or a module-qualified path, resolves through the import's target, with confidence `imported`; (4) otherwise the chain's last segment is looked up by name, yielding `unique_name` for one candidate, `ambiguous` for 2 to 5, `too_ambiguous` for more than 5, and `failed` for none. Candidates SHALL be limited to the reference's language family (Python; JavaScript and TypeScript; Rust).

#### Scenario: Method call on self

- **WHEN** a method calls `self.validate(user)` and its class defines `validate`
- **THEN** the reference resolves to that class's `validate` method with confidence `exact`

#### Scenario: Inherited member through self

- **WHEN** a method calls `self.save()`, its class defines no `save`, and its base class does
- **THEN** the reference resolves to the base class's `save` method with confidence `exact`

#### Scenario: Call to a function in the same file

- **WHEN** a function calls `helper()` and `helper` is defined at file scope in the same file
- **THEN** the reference resolves to that `helper` with confidence `exact`

#### Scenario: Innermost enclosing definition wins

- **WHEN** a nested function named `run` exists inside a function that calls `run()`, and a file-scope `run` also exists
- **THEN** the reference resolves to the nested `run`

#### Scenario: Method bodies do not see sibling methods by bare name

- **WHEN** a method calls bare `validate()`, its class defines a method `validate`, and nothing else named `validate` is in scope or imported
- **THEN** the reference does not resolve with confidence `exact`

#### Scenario: Call through an import

- **WHEN** a file contains `from pkg.auth import login` and calls `login()`, and `pkg/auth.py` defines `login`
- **THEN** the reference resolves to that `login` with confidence `imported`

#### Scenario: Aliased import

- **WHEN** a file contains `from pkg.auth import login as sign_in` and calls `sign_in()`
- **THEN** the reference resolves to `pkg/auth.py`'s `login` with confidence `imported`

#### Scenario: Module-qualified call

- **WHEN** a file contains `import pkg.auth` and calls `pkg.auth.login()`
- **THEN** the reference resolves to `pkg/auth.py`'s `login` with confidence `imported`

#### Scenario: Function-local import shadows a file-level one

- **WHEN** a file imports `load` from one module at file scope, a function imports `load` from another module, and that function calls `load()`
- **THEN** the call resolves through the function-local import

#### Scenario: Name unique in the repository

- **WHEN** a call's receiver is not bound by any tier and its last segment names exactly one function or method in the repository
- **THEN** the reference resolves to it with confidence `unique_name`

#### Scenario: Two to five candidates

- **WHEN** a call's last segment names three methods in the repository and no earlier tier applies
- **THEN** the reference is `ambiguous` with those three candidate IDs

#### Scenario: More than five candidates

- **WHEN** a call's last segment names eight methods in the repository and no earlier tier applies
- **THEN** the reference is `too_ambiguous` and produces no edges

#### Scenario: No match

- **WHEN** a call's name matches nothing by any tier
- **THEN** the reference is `failed`

#### Scenario: Resolution never crosses language families

- **WHEN** a Python call's name matches only a JavaScript function
- **THEN** the reference is `failed`

### Requirement: Name lookup is narrowed to the classes a file uses

Before a tier-4 lookup searches the whole repository, candidates SHALL be narrowed to members of the classes the reference's file defines or imports, together with those classes' resolved bases. Only when the narrowed set is empty SHALL the lookup search the repository. A narrowed lookup yielding exactly one candidate SHALL have confidence `unique_name`.

#### Scenario: Imported class disambiguates a method name

- **WHEN** five classes define `parse`, a test file imports only `RustParser`, and calls `parser.parse()` on an unbound receiver
- **THEN** the reference resolves to `RustParser.parse` with confidence `unique_name`

#### Scenario: Inherited method found through an imported class

- **WHEN** a file imports `RustParser`, which inherits `parse` from `BaseParser`, and calls `parser.parse()` on an unbound receiver
- **THEN** the reference resolves to `BaseParser.parse`

#### Scenario: Empty narrowed set falls back to the repository

- **WHEN** no class the file defines or imports has a member with the call's name, and exactly one method in the repository has it
- **THEN** the reference resolves to that method with confidence `unique_name`

### Requirement: Builtin-type method names are not guessed repository-wide

A repository-wide tier-4 lookup SHALL NOT be attempted for an attribute call on an unresolved receiver whose last segment is a method name of the language's builtin types (for example Python `get`, `append`, `items`, `join`; JavaScript `push`, `map`, `then`; Rust `unwrap`, `clone`, `iter`); such a reference SHALL be `failed`. Narrowed lookups SHALL still consider these names.

#### Scenario: Dictionary access is not guessed

- **WHEN** a function calls `config.get("key")` on an unbound receiver and one repository class defines `get`, but the file neither defines nor imports that class
- **THEN** the reference is `failed`

#### Scenario: Narrowed lookup still finds a same-named method

- **WHEN** a file imports `VectorStore`, which defines `add`, and calls `store.add(doc)` on an unbound receiver
- **THEN** the reference resolves to `VectorStore.add`

### Requirement: Python modules resolve by relative position or dotted suffix

A relative Python module (`.`, `..pkg`) SHALL resolve against the importing file's package directory. An absolute dotted module SHALL resolve to a repository file whose path ends, at a path-component boundary, with the dotted path as a `.py` file or as a package's `__init__.py`; when several files match, the one sharing the longest directory prefix with the importing file SHALL be chosen, then the shortest path. An imported name SHALL be tried first as a submodule, then as a top-level node of the module, then as a name the module itself imports (a re-export), following at most 5 re-export hops. When the module resolves but the name does not, the target SHALL be the module's file node. An absolute module matching no repository file SHALL be external, named by its first segment.

#### Scenario: Source layout needs no configuration

- **WHEN** `src/app/cli.py` contains `from app.config import load` and `src/app/config.py` defines `load`
- **THEN** the import resolves to that `load`

#### Scenario: Relative import

- **WHEN** `pkg/sub/a.py` contains `from ..util import helper` and `pkg/util.py` defines `helper`
- **THEN** the import resolves to that `helper`

#### Scenario: Package re-export is followed

- **WHEN** `pkg/__init__.py` contains `from .core import Engine`, `pkg/core.py` defines `Engine`, and another file contains `from pkg import Engine`
- **THEN** that import resolves to `pkg/core.py`'s `Engine`

#### Scenario: Imported submodule

- **WHEN** a file contains `from pkg import util` and `pkg/util.py` exists
- **THEN** the import resolves to `pkg/util.py`'s file node

#### Scenario: Name that is not a node

- **WHEN** a file contains `from pkg.settings import defaults` and `pkg/settings.py` exists but defines no node named `defaults`
- **THEN** the import resolves to `pkg/settings.py`'s file node

#### Scenario: Third-party module

- **WHEN** a file contains `from pydantic.fields import Field` and no repository file matches `pydantic/fields`
- **THEN** the import is `external` with target `external::pydantic`

### Requirement: JavaScript and TypeScript specifiers resolve by path

A specifier beginning with `./` or `../` SHALL resolve against the importing file's directory by trying the exact path, then the path with `.ts`, `.tsx`, `.js`, `.jsx`, `.mjs`, and `.cjs` appended, then an `index` file with each of those extensions inside it; a specifier ending in `.js`, `.jsx`, `.mjs`, or `.cjs` SHALL additionally be tried with the corresponding TypeScript extension. Any other specifier SHALL be external, named by its package: the first two segments for a scoped `@scope/name` specifier, otherwise the first segment. A `default` import SHALL resolve to the module's top-level node named like the binding, else the module's only top-level class or function, else the module's file node.

#### Scenario: Extension inference

- **WHEN** `src/app.ts` imports `{ render } from './view'` and `src/view.tsx` defines `render`
- **THEN** the import resolves to that `render`

#### Scenario: Directory index

- **WHEN** a file imports from `./components` and `components/index.ts` exists
- **THEN** the import resolves within `components/index.ts`

#### Scenario: TypeScript ESM `.js` specifier

- **WHEN** a TypeScript file imports from `./util.js` and only `util.ts` exists
- **THEN** the import resolves within `util.ts`

#### Scenario: Barrel re-export is followed

- **WHEN** `lib/index.ts` contains `export { Client } from './client'`, `lib/client.ts` defines `Client`, and another file imports `{ Client } from './lib'`
- **THEN** that import resolves to `lib/client.ts`'s `Client`

#### Scenario: Scoped package

- **WHEN** a file imports from `@tanstack/react-query/devtools`
- **THEN** the import is `external` with target `external::@tanstack/react-query`

#### Scenario: CommonJS require

- **WHEN** a file contains `const util = require('./util')` and calls `util.format()`, and `util.js` defines `format`
- **THEN** the call resolves to that `format` with confidence `imported`

### Requirement: Rust paths resolve through the module tree

A Rust file's module path SHALL be derived from its location under its crate root — the nearest ancestor directory containing `lib.rs` or `main.rs` — with `mod.rs`, `lib.rs`, and `main.rs` naming their directory's module. A path beginning with `crate` SHALL resolve from the crate root, `self` from the current module, and `super` from its parent; a path beginning with any other name SHALL resolve as a child module of the current module when one exists and otherwise be external, named by that first segment. Each module segment SHALL resolve to `<name>.rs` or `<name>/mod.rs`, and a final segment SHALL be tried as a submodule before as an item. A Rust type's members SHALL be the methods, in any Rust file, whose scope path is the type's name with or without a trait suffix.

#### Scenario: Crate-rooted use

- **WHEN** `src/main.rs` exists, `src/net/client.rs` contains `use crate::auth::Handler;`, and `src/auth/mod.rs` defines `Handler`
- **THEN** the import resolves to that `Handler`

#### Scenario: super path

- **WHEN** `src/net/client.rs` calls `super::retry()` and `src/net/mod.rs` defines `retry`
- **THEN** the call resolves to that `retry` with confidence `imported`

#### Scenario: Standard library

- **WHEN** a file contains `use std::collections::HashMap;`
- **THEN** the import is `external` with target `external::std`

#### Scenario: Associated function through a type

- **WHEN** a file imports `Handler` and calls `Handler::new()`, and `impl Handler { fn new() }` is in a different file from `struct Handler`
- **THEN** the call resolves to `Handler.new` with confidence `imported`

#### Scenario: Self inside a trait impl

- **WHEN** a method in `impl Display for Foo` calls `self.label()` and `impl Foo` defines `label`
- **THEN** the call resolves to `Foo.label` with confidence `exact`

### Requirement: Inheritance declared outside the deriving type resolves its source

A reference carrying a `for_type` SHALL resolve that type name with the same scope, import, and unique-name rules used for heads, restricted to type kinds, and the resolved type SHALL be the source of the reference's edge. When `for_type` cannot be resolved, the reference SHALL be `failed`.

#### Scenario: Trait implemented for a struct in another file

- **WHEN** `src/fmt.rs` contains `impl Display for crate::model::Foo` and `src/model.rs` defines `struct Foo`
- **THEN** the `inherits` edge's source is `src/model.rs::Foo#struct`

#### Scenario: Two trait impls for one type

- **WHEN** a file contains `impl Display for Foo` and `impl Debug for Foo`
- **THEN** two `inherits` references originate edges from `Foo`, one to each trait

### Requirement: Resolution is repeated whenever it runs

Every resolution run SHALL re-evaluate every reference, including those previously `failed`, `ambiguous`, or `too_ambiguous`, so that outcomes reflect the repository's current contents.

#### Scenario: Failed reference succeeds after a definition is added

- **WHEN** a call to `normalize()` was `failed`, and a file defining `normalize` is added and the repository synced
- **THEN** the call is `resolved`

#### Scenario: Unique name becomes ambiguous

- **WHEN** a reference resolved with `unique_name`, and a second function of the same name is added in another file and the repository synced
- **THEN** the reference is `ambiguous` with both candidates

#### Scenario: Target deleted

- **WHEN** a reference resolved to a function, and that function is deleted and the repository synced
- **THEN** the reference no longer names the deleted function as its target
