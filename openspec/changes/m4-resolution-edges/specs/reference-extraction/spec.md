## MODIFIED Requirements

### Requirement: References carry their origin and target text

Every reference SHALL carry the ID of the node it appears inside, the raw reference text as written, its head identifier, its kind, and the 1-indexed line and column where it appears. An import reference SHALL additionally carry the imported member name when it imports a member of a module, and an inheritance reference declared outside its deriving type SHALL additionally carry that type's name as `for_type`; both SHALL be absent otherwise. References SHALL NOT carry a resolved target — resolution happens later.

#### Scenario: A call inside a method

- **WHEN** a method body calls another function
- **THEN** the reference names the enclosing method's node ID as its origin, and carries the called text, its head, kind `calls`, and its line and column

#### Scenario: A reference at file scope

- **WHEN** a call appears at file scope, outside any symbol
- **THEN** its origin is the file node's ID

#### Scenario: References are unresolved when extracted

- **WHEN** any reference is extracted
- **THEN** it carries no resolved target and no confidence, leaving those for resolution

#### Scenario: Line and column locate the reference

- **WHEN** a reference is extracted
- **THEN** its line is 1-indexed and identifies the line the reference appears on

#### Scenario: Calls carry no import or implementation fields

- **WHEN** a call or a base-class reference inside its class is extracted
- **THEN** it carries no imported member name and no `for_type`

### Requirement: Attribute chains reduce to a head identifier

A call or inheritance reference written as a dotted, member, or path chain SHALL carry the full chain as its raw text and the leftmost base identifier as its head. When the base of a chain is not an identifier — for example a call, a literal, or a subscript — the reference SHALL be recorded with no head. Import references SHALL instead carry the name they bind as their head.

#### Scenario: Method call on self

- **WHEN** a method body contains `self.validate(user)`
- **THEN** the reference's raw text is `self.validate` and its head is `self`

#### Scenario: Deeply nested attribute chain

- **WHEN** a call is written `os.path.join(a, b)`
- **THEN** the reference's raw text is `os.path.join` and its head is `os`

#### Scenario: Bare function call

- **WHEN** a call is written `authenticate(user)`
- **THEN** the reference's raw text and head are both `authenticate`

#### Scenario: Chained call has no identifier head

- **WHEN** a call is written `build().run()`
- **THEN** the reference for `run` is recorded with no head

#### Scenario: Rust path call

- **WHEN** a Rust call is written `Handler::new()`
- **THEN** the reference's raw text is `Handler::new` and its head is `Handler`

### Requirement: Imports are recorded as references, per-language

Import statements SHALL be recorded as references of kind `imports`, one per name the statement binds, carrying the module or specifier text as written as raw text, the imported member (if any) as its imported name, and the bound local name (if any) as its head. A statement binding no name — a wildcard import or a side-effect-only import — SHALL yield one reference with no head, with imported name `*` for a wildcard. Relative-import markers in Python and specifier text in JavaScript and TypeScript SHALL be preserved as written. JavaScript and TypeScript `export … from` re-exports and `require()` calls SHALL also be recorded as imports; a `require()` assigned to a variable SHALL carry that variable as its head.

#### Scenario: Python plain import

- **WHEN** a Python file contains `import os.path`
- **THEN** one reference of kind `imports` is yielded with raw text `os.path`, no imported name, and head `os`

#### Scenario: Python aliased import

- **WHEN** a Python file contains `import numpy as np`
- **THEN** one reference is yielded with raw text `numpy`, no imported name, and head `np`

#### Scenario: Python from-import

- **WHEN** a Python file contains `from a.b import thing, other as alias`
- **THEN** two references are yielded, both with raw text `a.b`: one with imported name `thing` and head `thing`, and one with imported name `other` and head `alias`

#### Scenario: Python relative import

- **WHEN** a Python file contains `from . import sibling` or `from ..pkg import thing`
- **THEN** the references carry raw text `.` or `..pkg` respectively, preserving the leading-dot markers, with imported names `sibling` and `thing`

#### Scenario: Python wildcard import

- **WHEN** a Python file contains `from a import *`
- **THEN** one reference is yielded with raw text `a`, imported name `*`, and no head

#### Scenario: JavaScript default, named, and namespace imports

- **WHEN** a JavaScript or TypeScript file contains `import X, { a as b } from './m'` and `import * as ns from 'react'`
- **THEN** references are yielded with raw text `./m`, imported name `default`, head `X`; raw text `./m`, imported name `a`, head `b`; and raw text `react`, no imported name, head `ns`

#### Scenario: JavaScript side-effect import

- **WHEN** a JavaScript file contains `import './polyfill'`
- **THEN** one reference is yielded with raw text `./polyfill`, no imported name, and no head

#### Scenario: JavaScript require

- **WHEN** a JavaScript file contains `const util = require('./util')`
- **THEN** a reference of kind `imports` is yielded with raw text `./util` and head `util`

#### Scenario: TypeScript re-export

- **WHEN** a TypeScript file contains `export { Client as C } from './client'` and `export * from './types'`
- **THEN** references are yielded with raw text `./client`, imported name `Client`, head `C`; and raw text `./types`, imported name `*`, no head

#### Scenario: Rust use declaration

- **WHEN** a Rust file contains `use crate::auth::Handler;`
- **THEN** a reference of kind `imports` is yielded with raw text `crate::auth`, imported name `Handler`, and head `Handler`

#### Scenario: Rust grouped and aliased use

- **WHEN** a Rust file contains `use std::{io, fmt::Display as Show};`
- **THEN** references are yielded with raw text `std`, imported name `io`, head `io`; and raw text `std::fmt`, imported name `Display`, head `Show`

#### Scenario: Rust single-segment use

- **WHEN** a Rust file contains `use serde;`
- **THEN** a reference is yielded with raw text `serde`, no imported name, and head `serde`

### Requirement: Inheritance is recorded as references

Base classes, implemented interfaces, and Rust trait implementations SHALL be recorded as references of kind `inherits`. A reference declared inside the deriving type SHALL originate from that type. A Rust trait implementation SHALL carry the implemented trait as its raw text and the implementing type, as written, as its `for_type`.

#### Scenario: Python base class

- **WHEN** a Python class declares a base class
- **THEN** a reference of kind `inherits` is yielded from the subclass to the base class name

#### Scenario: Multiple bases

- **WHEN** a Python class declares two base classes
- **THEN** two references of kind `inherits` are yielded, one per base

#### Scenario: TypeScript extends and implements

- **WHEN** a TypeScript class extends a class and implements an interface
- **THEN** a reference of kind `inherits` is yielded for each

#### Scenario: Rust trait implementation

- **WHEN** a Rust file contains `impl Display for Foo`
- **THEN** a reference of kind `inherits` is yielded with raw text `Display` and `for_type` `Foo`

#### Scenario: Rust inherent impl

- **WHEN** a Rust file contains `impl Foo` with no trait
- **THEN** no `inherits` reference is yielded for it

## ADDED Requirements

### Requirement: Builtin references are dropped unless shadowed

A call or inheritance reference whose head is a builtin name of its language SHALL NOT be yielded, unless the same file defines a node with that name or binds that name by an import. The builtin names SHALL be fixed lists that do not depend on the running interpreter: Python's builtins, JavaScript and TypeScript global objects and functions, and Rust prelude names. Import references SHALL never be dropped.

#### Scenario: Python builtin call dropped

- **WHEN** a Python function calls `len(items)` and `isinstance(x, str)`
- **THEN** no reference is yielded for either call

#### Scenario: Builtin receiver dropped

- **WHEN** a JavaScript function calls `console.log(message)`
- **THEN** no reference is yielded for that call

#### Scenario: Rust prelude constructor dropped

- **WHEN** a Rust function calls `Some(value)` and `Vec::new()`
- **THEN** no reference is yielded for either call

#### Scenario: Builtin base class dropped

- **WHEN** a Python class declares `Exception` as its base
- **THEN** no `inherits` reference is yielded for it

#### Scenario: Locally defined name shadows a builtin

- **WHEN** a Python file defines a function named `open` and calls `open(path)`
- **THEN** a `calls` reference to `open` is yielded

#### Scenario: Imported name shadows a builtin

- **WHEN** a Python file contains `from .compat import filter` and calls `filter(items)`
- **THEN** a `calls` reference to `filter` is yielded
