"""Whole-output snapshots of composed text over every M2 fixture file.

Where `test_compose.py`'s hand-written tests assert specific facts (section
order, a per-kind variant, truncation), this asserts nothing goes
unaccounted for: every node's full composed output is captured, so a
composer format change shows up as a diff here even when it doesn't happen
to break a narrower test.

Snapshot values are populated by running:
    uv run --group test pytest --inline-snapshot=create src/indexter/index/tests/test_compose_snapshots.py
and updated (after an intentional change -- bump INDEX_FORMAT_VERSION first)
with `--inline-snapshot=fix`.
"""

from __future__ import annotations

from pathlib import Path

from inline_snapshot import snapshot

from indexter.index.compose import compose_file
from indexter.index.embed import FakeEmbedder
from indexter.parse.base import parse_file

FIXTURES = Path(__file__).parent.parent.parent / "parse" / "tests" / "fixtures"
TOKENIZER = FakeEmbedder().tokenizer()


def _render_composed(relpath: str) -> str:
    content = (FIXTURES / relpath).read_text()
    result = parse_file(relpath, content)
    composed = compose_file(relpath, content, result, TOKENIZER, 256)
    lines = []
    for node in sorted(result.nodes, key=lambda n: (n.start_byte, n.end_byte, n.kind.value, n.name)):
        c = composed[node.id]
        lines.append(f"### {node.kind.value} name={node.name!r} scope={node.scope_path}")
        lines.append(f"qualified_name={c.qualified_name!r}")
        lines.append(f"name_words={c.name_words!r}")
        lines.append("embed_text:")
        lines.append(c.embed_text)
        lines.append("body:")
        lines.append(c.body)
    return "\n".join(lines)


class TestSnapshot:
    def test_python_sample(self):
        assert _render_composed("python/sample.py") == snapshot('''\
### file name='' scope=()
qualified_name='python/sample.py'
name_words='python sample'
embed_text:
file python/sample.py | python sample | python/sample.py (python sample)
"""Sample module exercising imports, a documented class, decorators, and calls."""
constants: MAX_RETRIES
classes: Base, Handler
functions: deprecated, standalone

"""Sample module exercising imports, a documented class, decorators, and calls."""

import os
from collections import OrderedDict
from . import sibling
from ..pkg import thing
body:
"""Sample module exercising imports, a documented class, decorators, and calls."""

import os
from collections import OrderedDict
from . import sibling
from ..pkg import thing
### constant name='MAX_RETRIES' scope=()
qualified_name='MAX_RETRIES'
name_words='max retries'
embed_text:
constant MAX_RETRIES | max retries | python/sample.py (python sample)
MAX_RETRIES = 3
body:
MAX_RETRIES = 3
### class name='Base' scope=()
qualified_name='Base'
name_words='base'
embed_text:
class Base | base | python/sample.py (python sample)
class Base:
A base class with nothing interesting in it.
body:
class Base:
    """A base class with nothing interesting in it."""
### class name='Handler' scope=()
qualified_name='Handler'
name_words='handler'
embed_text:
class Handler | handler | python/sample.py (python sample)
class Handler(Base):
Handles requests, delegating validation to a helper.
methods: login, validate
body:
class Handler(Base):
    """Handles requests, delegating validation to a helper."""
### method name='login' scope=('Handler',)
qualified_name='Handler.login'
name_words='handler login'
embed_text:
method Handler.login | handler login | python/sample.py (python sample)
def login(self, user)
Authenticate a user and record the attempt.
self.validate(user)
        os.path.join("var", "log")
        return thing(user)
body:
def login(self, user):
        """Authenticate a user and record the attempt."""
        self.validate(user)
        os.path.join("var", "log")
        return thing(user)
### method name='validate' scope=('Handler',)
qualified_name='Handler.validate'
name_words='handler validate'
embed_text:
method Handler.validate | handler validate | python/sample.py (python sample)
def validate(self, user)
return user is not None
body:
def validate(self, user):
        return user is not None
### function name='deprecated' scope=()
qualified_name='deprecated'
name_words='deprecated'
embed_text:
function deprecated | deprecated | python/sample.py (python sample)
def deprecated(fn)
A fixture decorator -- not real, just wraps its argument.
return fn
body:
def deprecated(fn):
    """A fixture decorator -- not real, just wraps its argument."""
    return fn
### function name='standalone' scope=()
qualified_name='standalone'
name_words='standalone'
embed_text:
function standalone | standalone | python/sample.py (python sample)
def standalone()
A decorated module-level function.
return MAX_RETRIES
body:
@deprecated
def standalone():
    """A decorated module-level function."""
    return MAX_RETRIES\
''')

    def test_python_scopes(self):
        assert _render_composed("python/scopes.py") == snapshot('''\
### file name='' scope=()
qualified_name='python/scopes.py'
name_words='python scopes'
embed_text:
file python/scopes.py | python scopes | python/scopes.py (python scopes)
"""Nested-function and duplicate-name collision cases for node identity."""
functions: outer_a, outer_b, login, login

"""Nested-function and duplicate-name collision cases for node identity."""
body:
"""Nested-function and duplicate-name collision cases for node identity."""
### function name='outer_a' scope=()
qualified_name='outer_a'
name_words='outer a'
embed_text:
function outer_a | outer a | python/scopes.py (python scopes)
def outer_a()
return inner
body:
def outer_a():
    \n\

    return inner
### function name='inner' scope=('outer_a',)
qualified_name='outer_a.inner'
name_words='outer a inner'
embed_text:
function outer_a.inner | outer a inner | python/scopes.py (python scopes)
def inner()
return "a"
body:
def inner():
        return "a"
### function name='outer_b' scope=()
qualified_name='outer_b'
name_words='outer b'
embed_text:
function outer_b | outer b | python/scopes.py (python scopes)
def outer_b()
return inner
body:
def outer_b():
    \n\

    return inner
### function name='inner' scope=('outer_b',)
qualified_name='outer_b.inner'
name_words='outer b inner'
embed_text:
function outer_b.inner | outer b inner | python/scopes.py (python scopes)
def inner()
return "b"
body:
def inner():
        return "b"
### function name='login' scope=()
qualified_name='login'
name_words='login'
embed_text:
function login | login | python/scopes.py (python scopes)
def login()
return "first"
body:
def login():
    return "first"
### function name='login' scope=()
qualified_name='login'
name_words='login'
embed_text:
function login | login | python/scopes.py (python scopes)
def login():
    # Deliberate duplicate name at file scope -- exercises the ~N suffix rule.
return "second"
body:
def login():
    # Deliberate duplicate name at file scope -- exercises the ~N suffix rule.
    return "second"\
''')

    def test_javascript_sample(self):
        assert _render_composed("javascript/sample.js") == snapshot("""\
### file name='' scope=()
qualified_name='javascript/sample.js'
name_words='javascript sample'
embed_text:
file javascript/sample.js | javascript sample | javascript/sample.js (javascript sample)
import { EventEmitter } from "events";
classes: Base, Handler
functions: standalone, double

import { EventEmitter } from "events";
import defaultExport from "./utils";
body:
import { EventEmitter } from "events";
import defaultExport from "./utils";
### class name='Base' scope=()
qualified_name='Base'
name_words='base'
embed_text:
class Base | base | javascript/sample.js (javascript sample)
class Base extends EventEmitter {}
body:
class Base extends EventEmitter {}
### class name='Handler' scope=()
qualified_name='Handler'
name_words='handler'
embed_text:
class Handler | handler | javascript/sample.js (javascript sample)
class Handler extends Base {
methods: process
body:
class Handler extends Base {
  \n\
}
### method name='process' scope=('Handler',)
qualified_name='Handler.process'
name_words='handler process'
embed_text:
method Handler.process | handler process | javascript/sample.js (javascript sample)
process(items)
{
    items.forEach();
  }
body:
process(items) {
    items.forEach();
  }
### function name='onItem' scope=('Handler', 'process')
qualified_name='Handler.process.onItem'
name_words='handler process on item'
embed_text:
function Handler.process.onItem | handler process on item | javascript/sample.js (javascript sample)
function onItem(item)
{
      return item;
    }
body:
function onItem(item) {
      return item;
    }
### function name='standalone' scope=()
qualified_name='standalone'
name_words='standalone'
embed_text:
function standalone | standalone | javascript/sample.js (javascript sample)
function standalone(x)
{
  return x * 2;
}
body:
function standalone(x) {
  return x * 2;
}
### function name='double' scope=()
qualified_name='double'
name_words='double'
embed_text:
function double | double | javascript/sample.js (javascript sample)
(x) =>
const double = (x) => x * 2;
body:
const double = (x) => x * 2;\
""")

    def test_javascript_objects(self):
        assert _render_composed("javascript/objects.js") == snapshot("""\
### file name='' scope=()
qualified_name='javascript/objects.js'
name_words='javascript objects'
embed_text:
file javascript/objects.js | javascript objects | javascript/objects.js (javascript objects)
const first = {
methods: handler, handler

const first = {
  ,
};

const second = {
  ,
};
body:
const first = {
  ,
};

const second = {
  ,
};
### method name='handler' scope=('first',)
qualified_name='first.handler'
name_words='first handler'
embed_text:
method first.handler | first handler | javascript/objects.js (javascript objects)
handler(event)
{
    return event.type;
  }
body:
handler(event) {
    return event.type;
  }
### method name='handler' scope=('second',)
qualified_name='second.handler'
name_words='second handler'
embed_text:
method second.handler | second handler | javascript/objects.js (javascript objects)
handler(event)
{
    return event.target;
  }
body:
handler(event) {
    return event.target;
  }\
""")

    def test_typescript_sample(self):
        assert _render_composed("typescript/sample.ts") == snapshot("""\
### file name='' scope=()
qualified_name='typescript/sample.ts'
name_words='typescript sample'
embed_text:
file typescript/sample.ts | typescript sample | typescript/sample.ts (typescript sample)
export interface Greeter {
interfaces: Greeter
type aliases: Level
enums: Status
classes: BaseHandler, Handler
functions: standalone

export \n\

export \n\

export \n\



export \n\

export
body:
export \n\

export \n\

export \n\



export \n\

export
### interface name='Greeter' scope=()
qualified_name='Greeter'
name_words='greeter'
embed_text:
interface Greeter | greeter | typescript/sample.ts (typescript sample)
interface Greeter {
methods: greet
body:
interface Greeter {
  ;
}
### method name='greet' scope=('Greeter',)
qualified_name='Greeter.greet'
name_words='greeter greet'
embed_text:
method Greeter.greet | greeter greet | typescript/sample.ts (typescript sample)
greet(name: string): string
body:
greet(name: string): string
### type_alias name='Level' scope=()
qualified_name='Level'
name_words='level'
embed_text:
type_alias Level | level | typescript/sample.ts (typescript sample)
type Level = "low" | "medium" | "high";
body:
type Level = "low" | "medium" | "high";
### enum name='Status' scope=()
qualified_name='Status'
name_words='status'
embed_text:
enum Status | status | typescript/sample.ts (typescript sample)
enum Status {
body:
enum Status {
  Active,
  Inactive,
}
### class name='BaseHandler' scope=()
qualified_name='BaseHandler'
name_words='base handler'
embed_text:
class BaseHandler | base handler | typescript/sample.ts (typescript sample)
class BaseHandler {
methods: setup
body:
class BaseHandler {
  \n\
}
### method name='setup' scope=('BaseHandler',)
qualified_name='BaseHandler.setup'
name_words='base handler setup'
embed_text:
method BaseHandler.setup | base handler setup | typescript/sample.ts (typescript sample)
protected setup(): void
{}
body:
protected setup(): void {}
### class name='Handler' scope=()
qualified_name='Handler'
name_words='handler'
embed_text:
class Handler | handler | typescript/sample.ts (typescript sample)
class Handler extends BaseHandler implements Greeter {
methods: greet
body:
class Handler extends BaseHandler implements Greeter {
  \n\
}
### method name='greet' scope=('Handler',)
qualified_name='Handler.greet'
name_words='handler greet'
embed_text:
method Handler.greet | handler greet | typescript/sample.ts (typescript sample)
greet(name: string): string
{
    return `hello ${name}`;
  }
body:
greet(name: string): string {
    return `hello ${name}`;
  }
### function name='standalone' scope=()
qualified_name='standalone'
name_words='standalone'
embed_text:
function standalone | standalone | typescript/sample.ts (typescript sample)
function standalone(x: number): number
{
  return x * 2;
}
body:
function standalone(x: number): number {
  return x * 2;
}\
""")

    def test_rust_sample(self):
        assert _render_composed("rust/sample.rs") == snapshot("""\
### file name='' scope=()
qualified_name='rust/sample.rs'
name_words='rust sample'
embed_text:
file rust/sample.rs | rust sample | rust/sample.rs (rust sample)
use std::fmt;
structs: Foo
methods: new, fmt, fmt, greet
traits: Greet

use std::fmt;
use crate::helpers::assist;



impl Foo {
    \n\
}

impl fmt::Display for Foo {
    \n\
}

impl fmt::Debug for Foo {
    \n\
}



impl Greet for Foo {
    \n\
}
body:
use std::fmt;
use crate::helpers::assist;



impl Foo {
    \n\
}

impl fmt::Display for Foo {
    \n\
}

impl fmt::Debug for Foo {
    \n\
}



impl Greet for Foo {
    \n\
}
### struct name='Foo' scope=()
qualified_name='Foo'
name_words='foo'
embed_text:
struct Foo | foo | rust/sample.rs (rust sample)
pub struct Foo {
body:
pub struct Foo {
    pub value: i32,
}
### method name='new' scope=('Foo',)
qualified_name='Foo.new'
name_words='foo new'
embed_text:
method Foo.new | foo new | rust/sample.rs (rust sample)
pub fn new(value: i32) -> Foo
{
        Foo { value }
    }
body:
pub fn new(value: i32) -> Foo {
        Foo { value }
    }
### method name='fmt' scope=('Foo<Display>',)
qualified_name='Foo<Display>.fmt'
name_words='foo < display > fmt'
embed_text:
method Foo<Display>.fmt | foo < display > fmt | rust/sample.rs (rust sample)
fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
{
        write!(f, "Foo({})", self.value)
    }
body:
fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Foo({})", self.value)
    }
### method name='fmt' scope=('Foo<Debug>',)
qualified_name='Foo<Debug>.fmt'
name_words='foo < debug > fmt'
embed_text:
method Foo<Debug>.fmt | foo < debug > fmt | rust/sample.rs (rust sample)
fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result
{
        write!(f, "Foo {{ value: {} }}", self.value)
    }
body:
fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Foo {{ value: {} }}", self.value)
    }
### trait name='Greet' scope=()
qualified_name='Greet'
name_words='greet'
embed_text:
trait Greet | greet | rust/sample.rs (rust sample)
pub trait Greet {
body:
pub trait Greet {
    fn greet(&self) -> String;
}
### method name='greet' scope=('Foo<Greet>',)
qualified_name='Foo<Greet>.greet'
name_words='foo < greet > greet'
embed_text:
method Foo<Greet>.greet | foo < greet > greet | rust/sample.rs (rust sample)
fn greet(&self) -> String
{
        assist(self.value)
    }
body:
fn greet(&self) -> String {
        assist(self.value)
    }\
""")

    def test_css_sample(self):
        assert _render_composed("css/sample.css") == snapshot("""\
### section name='body' scope=()
qualified_name='body'
name_words='body'
embed_text:
section body | body | css/sample.css (css sample)
rule_set
body {
  margin: 0;
  font-family: sans-serif;
}
body:
body {
  margin: 0;
  font-family: sans-serif;
}
### file name='' scope=()
qualified_name='css/sample.css'
name_words='css sample'
embed_text:
file css/sample.css | css sample | css/sample.css (css sample)
body {
sections: body, @media
body:

### section name='@media' scope=()
qualified_name='@media'
name_words='@media'
embed_text:
section @media | @media | css/sample.css (css sample)
media_statement
@media (max-width: 600px) {
  \n\
}
body:
@media (max-width: 600px) {
  \n\
}
### section name='body' scope=('@media',)
qualified_name='@media.body'
name_words='@media body'
embed_text:
section @media.body | @media body | css/sample.css (css sample)
rule_set
body {
    font-size: 14px;
  }
body:
body {
    font-size: 14px;
  }\
""")

    def test_json_sample(self):
        assert _render_composed("json/sample.json") == snapshot("""\
### data name='' scope=()
qualified_name='json/sample.json'
name_words=''
embed_text:
data json/sample.json |  | json/sample.json (json sample)
object
{
  "name": "sample",
  "version": 1,
  "settings": \n\
}
body:
{
  "name": "sample",
  "version": 1,
  "settings": \n\
}
### file name='' scope=()
qualified_name='json/sample.json'
name_words='json sample'
embed_text:
file json/sample.json | json sample | json/sample.json (json sample)
{
data: (anonymous)
body:

### data name='settings' scope=()
qualified_name='settings'
name_words='settings'
embed_text:
data settings | settings | json/sample.json (json sample)
object
{
    "enabled": true,
    "limits": \n\
  }
body:
{
    "enabled": true,
    "limits": \n\
  }
### data name='limits' scope=('settings',)
qualified_name='settings.limits'
name_words='settings limits'
embed_text:
data settings.limits | settings limits | json/sample.json (json sample)
array
[1, 2, 3]
body:
[1, 2, 3]\
""")

    def test_toml_sample(self):
        assert _render_composed("toml/sample.toml") == snapshot("""\
### data name='name' scope=()
qualified_name='name'
name_words='name'
embed_text:
data name | name | toml/sample.toml (toml sample)
pair
name = "sample"
body:
name = "sample"
### file name='' scope=()
qualified_name='toml/sample.toml'
name_words='toml sample'
embed_text:
file toml/sample.toml | toml sample | toml/sample.toml (toml sample)
name = "sample"
data: name, version, settings, items, items
body:

### data name='version' scope=()
qualified_name='version'
name_words='version'
embed_text:
data version | version | toml/sample.toml (toml sample)
pair
version = 1
body:
version = 1
### data name='settings' scope=()
qualified_name='settings'
name_words='settings'
embed_text:
data settings | settings | toml/sample.toml (toml sample)
table
[settings]
enabled = true
limits = [1, 2, 3]
body:
[settings]
enabled = true
limits = [1, 2, 3]
### data name='items' scope=()
qualified_name='items'
name_words='items'
embed_text:
data items | items | toml/sample.toml (toml sample)
table_array_element
[[items]]
id = 1
body:
[[items]]
id = 1
### data name='items' scope=()
qualified_name='items'
name_words='items'
embed_text:
data items | items | toml/sample.toml (toml sample)
table_array_element
[[items]]
id = 2
body:
[[items]]
id = 2\
""")

    def test_markdown_sample(self):
        assert _render_composed("markdown/sample.md") == snapshot("""\
### file name='' scope=()
qualified_name='markdown/sample.md'
name_words='markdown sample'
embed_text:
file markdown/sample.md | markdown sample | markdown/sample.md (markdown sample)
# Indexter Fixture
sections: Indexter Fixture > Setup, Indexter Fixture > Usage

# Indexter Fixture

Top-level introduction paragraph.
body:
# Indexter Fixture

Top-level introduction paragraph.
### section name='Indexter Fixture' scope=()
qualified_name='Indexter Fixture'
name_words='indexter fixture'
embed_text:
section Indexter Fixture | indexter fixture | markdown/sample.md (markdown sample)
h1
# Indexter Fixture

Top-level introduction paragraph.

## Setup

Explains how to set things up.

### Prerequisites

A nested list of things you need first.

## Usage

### Basic usage

Some basic usage text.

### Advanced usage

Some advanced usage text.
body:
# Indexter Fixture

Top-level introduction paragraph.

## Setup

Explains how to set things up.

### Prerequisites

A nested list of things you need first.

## Usage

### Basic usage

Some basic usage text.

### Advanced usage

Some advanced usage text.
### section name='Indexter Fixture > Setup' scope=()
qualified_name='Indexter Fixture > Setup'
name_words='indexter fixture > setup'
embed_text:
section Indexter Fixture > Setup | indexter fixture > setup | markdown/sample.md (markdown sample)
h2
## Setup

Explains how to set things up.
body:
## Setup

Explains how to set things up.
### section name='Indexter Fixture > Setup > Prerequisites' scope=()
qualified_name='Indexter Fixture > Setup > Prerequisites'
name_words='indexter fixture > setup > prerequisites'
embed_text:
section Indexter Fixture > Setup > Prerequisites | indexter fixture > setup > prerequisites | markdown/sample.md (markdown sample)
h3
### Prerequisites

A nested list of things you need first.
body:
### Prerequisites

A nested list of things you need first.
### section name='Indexter Fixture > Usage' scope=()
qualified_name='Indexter Fixture > Usage'
name_words='indexter fixture > usage'
embed_text:
section Indexter Fixture > Usage | indexter fixture > usage | markdown/sample.md (markdown sample)
h2
## Usage
body:
## Usage
### section name='Indexter Fixture > Usage > Basic usage' scope=()
qualified_name='Indexter Fixture > Usage > Basic usage'
name_words='indexter fixture > usage > basic usage'
embed_text:
section Indexter Fixture > Usage > Basic usage | indexter fixture > usage > basic usage | markdown/sample.md (markdown sample)
h3
### Basic usage

Some basic usage text.
body:
### Basic usage

Some basic usage text.
### section name='Indexter Fixture > Usage > Advanced usage' scope=()
qualified_name='Indexter Fixture > Usage > Advanced usage'
name_words='indexter fixture > usage > advanced usage'
embed_text:
section Indexter Fixture > Usage > Advanced usage | indexter fixture > usage > advanced usage | markdown/sample.md (markdown sample)
h3
### Advanced usage

Some advanced usage text.
body:
### Advanced usage

Some advanced usage text.\
""")

    def test_html_sample(self):
        assert _render_composed("html/sample.html") == snapshot("""\
### file name='' scope=()
qualified_name='html/sample.html'
name_words='html sample'
embed_text:
file html/sample.html | html sample | html/sample.html (html sample)
<!doctype html>
sections: Sample Page, table, ul-list

<!doctype html>
<html>
  <head>
    <title>Sample</title>
  </head>
  <body>
    \n\
    \n\
    \n\
  </body>
</html>
body:
<!doctype html>
<html>
  <head>
    <title>Sample</title>
  </head>
  <body>
    \n\
    \n\
    \n\
  </body>
</html>
### section name='Sample Page' scope=()
qualified_name='Sample Page'
name_words='sample page'
embed_text:
section Sample Page | sample page | html/sample.html (html sample)
h1
<h1>Sample Page</h1>
body:
<h1>Sample Page</h1>
### section name='table' scope=()
qualified_name='table'
name_words='table'
embed_text:
section table | table | html/sample.html (html sample)
table
<table>
      <tr><td>1</td><td>2</td></tr>
    </table>
body:
<table>
      <tr><td>1</td><td>2</td></tr>
    </table>
### section name='ul-list' scope=()
qualified_name='ul-list'
name_words='ul list'
embed_text:
section ul-list | ul list | html/sample.html (html sample)
ul
<ul>
      <li>one</li>
      <li>two</li>
    </ul>
body:
<ul>
      <li>one</li>
      <li>two</li>
    </ul>\
""")

    def test_yaml_sample(self):
        assert _render_composed("yaml/sample.yaml") == snapshot("""\
### data name='' scope=()
qualified_name='yaml/sample.yaml'
name_words=''
embed_text:
data yaml/sample.yaml |  | yaml/sample.yaml (yaml sample)
block_mapping
name: sample
version: 1
settings:
  enabled: true
  limits:
    - 1
    - 2
    - 3
body:
name: sample
version: 1
settings:
  enabled: true
  limits:
    - 1
    - 2
    - 3
### file name='' scope=()
qualified_name='yaml/sample.yaml'
name_words='yaml sample'
embed_text:
file yaml/sample.yaml | yaml sample | yaml/sample.yaml (yaml sample)
name: sample
data: settings

name: sample
version: 1
settings:
body:
name: sample
version: 1
settings:
### data name='settings' scope=()
qualified_name='settings'
name_words='settings'
embed_text:
data settings | settings | yaml/sample.yaml (yaml sample)
block_mapping
enabled: true
  limits:
body:
enabled: true
  limits:
### data name='limits' scope=('settings',)
qualified_name='settings.limits'
name_words='settings limits'
embed_text:
data settings.limits | settings limits | yaml/sample.yaml (yaml sample)
block_sequence
- 1
    - 2
    - 3
body:
- 1
    - 2
    - 3\
""")

    def test_other_fallback_chunking(self):
        assert _render_composed("other/sample.xyz") == snapshot("""\
### chunk name='' scope=()
qualified_name='other/sample.xyz:1-6'
name_words='other sample'
embed_text:
other/sample.xyz:1-6
This is a plain-text file with an extension indexter doesn't recognize.
It should fall back to the chunking parser rather than any language parser.
Repeat this sentence a few times to make sure it spans more than one chunk
in the snapshot tests that will use a small chunk size. Repeat this sentence
a few times to make sure it spans more than one chunk in the snapshot tests.

body:
This is a plain-text file with an extension indexter doesn't recognize.
It should fall back to the chunking parser rather than any language parser.
Repeat this sentence a few times to make sure it spans more than one chunk
in the snapshot tests that will use a small chunk size. Repeat this sentence
a few times to make sure it spans more than one chunk in the snapshot tests.
### file name='' scope=()
qualified_name='other/sample.xyz'
name_words='other sample'
embed_text:
file other/sample.xyz | other sample | other/sample.xyz (other sample)
This is a plain-text file with an extension indexter doesn't recognize.
This is a plain-text file with an extension indexter doesn't recognize.
It should fall back to the chunking parser rather than any language parser.
Repeat this sentence a few times to make sure it spans more than one chunk
in the snapshot tests that will use a small chunk size. Repeat this sentence
a few times to make sure it spans more than one chunk in the snapshot tests.
body:
This is a plain-text file with an extension indexter doesn't recognize.
It should fall back to the chunking parser rather than any language parser.
Repeat this sentence a few times to make sure it spans more than one chunk
in the snapshot tests that will use a small chunk size. Repeat this sentence
a few times to make sure it spans more than one chunk in the snapshot tests.\
""")
