"""Whole-output snapshots over the fixture repo. Where the hand-written
tests in each `test_<language>.py` assert specific facts (a collision is
resolved, a reference's head is computed correctly), these assert nothing
goes unaccounted for: every node and every reference a fixture produces is
captured, so a stray extra node or a silently dropped reference shows up as
a diff instead of passing unnoticed.

Snapshot values are populated by running:
    uv run --group test pytest --inline-snapshot=create src/indexter/parse/tests/test_snapshots.py
and updated (after an intentional change) with `--inline-snapshot=fix`.
"""

from pathlib import Path

from inline_snapshot import snapshot

from indexter.config import Settings
from indexter.parse.base import parse_file
from indexter.parse.chunk import ChunkParser
from indexter.parse.tests.render import render_result

FIXTURES = Path(__file__).parent / "fixtures"


def _parse(relpath: str) -> str:
    content = (FIXTURES / relpath).read_text()
    return render_result(parse_file(relpath, content))


class TestCodeFixtures:
    def test_python_sample(self):
        assert _parse("python/sample.py") == snapshot("""\
NODES:
file name='' scope=() lang=python lines=1-37 bytes=0-791 sig=None doc=None
constant name='MAX_RETRIES' scope=() lang=python lines=8-8 bytes=177-192 sig=None doc=None
class name='Base' scope=() lang=python lines=11-12 bytes=195-261 sig=None doc='A base class with nothing interesting in it.'
class name='Handler' scope=() lang=python lines=15-25 bytes=264-586 sig=None doc='Handles requests, delegating validation to a helper.'
method name='login' scope=('Handler',) lang=python lines=18-22 bytes=353-523 sig='def login(self, user)' doc='Authenticate a user and record the attempt.'
method name='validate' scope=('Handler',) lang=python lines=24-25 bytes=529-586 sig='def validate(self, user)' doc=None
function name='deprecated' scope=() lang=python lines=28-30 bytes=589-690 sig='def deprecated(fn)' doc='A fixture decorator -- not real, just wraps its argument.'
function name='standalone' scope=() lang=python lines=33-36 bytes=693-790 sig='def standalone()' doc='A decorated module-level function.'

REFS:
imports raw='os' head='os' imported=None for_type=None line=3 col=8
imports raw='collections' head='OrderedDict' imported='OrderedDict' for_type=None line=4 col=25
imports raw='.' head='sibling' imported='sibling' for_type=None line=5 col=15
imports raw='..pkg' head='thing' imported='thing' for_type=None line=6 col=19
inherits raw='Base' head='Base' imported=None for_type=None line=15 col=15
calls raw='self.validate' head='self' imported=None for_type=None line=20 col=9
calls raw='os.path.join' head='os' imported=None for_type=None line=21 col=9
calls raw='thing' head='thing' imported=None for_type=None line=22 col=16\
""")

    def test_python_scopes(self):
        assert _parse("python/scopes.py") == snapshot("""\
NODES:
file name='' scope=() lang=python lines=1-25 bytes=0-366 sig=None doc=None
function name='outer_a' scope=() lang=python lines=4-8 bytes=78-146 sig='def outer_a()' doc=None
function name='inner' scope=('outer_a',) lang=python lines=5-6 bytes=97-128 sig='def inner()' doc=None
function name='outer_b' scope=() lang=python lines=11-15 bytes=149-217 sig='def outer_b()' doc=None
function name='inner' scope=('outer_b',) lang=python lines=12-13 bytes=168-199 sig='def inner()' doc=None
function name='login' scope=() lang=python lines=18-19 bytes=220-251 sig='def login()' doc=None
function name='login' scope=() lang=python lines=22-24 bytes=254-365 sig='def login():\\n    # Deliberate duplicate name at file scope -- exercises the ~N suffix rule.' doc=None

REFS:
""")

    def test_javascript_sample(self):
        assert _parse("javascript/sample.js") == snapshot("""\
NODES:
file name='' scope=() lang=javascript lines=1-19 bytes=0-310 sig=None doc=None
class name='Base' scope=() lang=javascript lines=4-4 bytes=77-111 sig=None doc=None
class name='Handler' scope=() lang=javascript lines=6-12 bytes=113-235 sig=None doc=None
method name='process' scope=('Handler',) lang=javascript lines=7-11 bytes=144-233 sig='process(items)' doc=None
function name='onItem' scope=('Handler', 'process') lang=javascript lines=8-10 bytes=179-227 sig='function onItem(item)' doc=None
function name='standalone' scope=() lang=javascript lines=14-16 bytes=237-279 sig='function standalone(x)' doc=None
function name='double' scope=() lang=javascript lines=18-18 bytes=281-309 sig='(x) =>' doc=None

REFS:
imports raw='events' head='EventEmitter' imported='EventEmitter' for_type=None line=1 col=10
imports raw='./utils' head='defaultExport' imported='default' for_type=None line=2 col=8
inherits raw='EventEmitter' head='EventEmitter' imported=None for_type=None line=4 col=20
inherits raw='Base' head='Base' imported=None for_type=None line=6 col=23
calls raw='items.forEach' head='items' imported=None for_type=None line=8 col=5\
""")

    def test_javascript_objects(self):
        assert _parse("javascript/objects.js") == snapshot("""\
NODES:
file name='' scope=() lang=javascript lines=1-12 bytes=0-136 sig=None doc=None
method name='handler' scope=('first',) lang=javascript lines=2-4 bytes=18-61 sig='handler(event)' doc=None
method name='handler' scope=('second',) lang=javascript lines=8-10 bytes=86-131 sig='handler(event)' doc=None

REFS:
""")

    def test_typescript_sample(self):
        assert _parse("typescript/sample.ts") == snapshot("""\
NODES:
file name='' scope=() lang=typescript lines=1-25 bytes=0-402 sig=None doc=None
interface name='Greeter' scope=() lang=typescript lines=1-3 bytes=7-59 sig=None doc=None
method name='greet' scope=('Greeter',) lang=typescript lines=2-2 bytes=29-56 sig='greet(name: string): string' doc=None
type_alias name='Level' scope=() lang=typescript lines=5-5 bytes=68-107 sig=None doc=None
enum name='Status' scope=() lang=typescript lines=7-10 bytes=116-153 sig=None doc=None
class name='BaseHandler' scope=() lang=typescript lines=12-14 bytes=155-205 sig=None doc=None
method name='setup' scope=('BaseHandler',) lang=typescript lines=13-13 bytes=177-203 sig='protected setup(): void' doc=None
class name='Handler' scope=() lang=typescript lines=16-20 bytes=214-334 sig=None doc=None
method name='greet' scope=('Handler',) lang=typescript lines=17-19 bytes=271-332 sig='greet(name: string): string' doc=None
function name='standalone' scope=() lang=typescript lines=22-24 bytes=343-401 sig='function standalone(x: number): number' doc=None

REFS:
inherits raw='BaseHandler' head='BaseHandler' imported=None for_type=None line=16 col=30
inherits raw='Greeter' head='Greeter' imported=None for_type=None line=16 col=53\
""")

    def test_rust_sample(self):
        assert _parse("rust/sample.rs") == snapshot("""\
NODES:
file name='' scope=() lang=rust lines=1-35 bytes=0-585 sig=None doc=None
struct name='Foo' scope=() lang=rust lines=4-6 bytes=43-81 sig=None doc=None
method name='new' scope=('Foo',) lang=rust lines=9-11 bytes=98-157 sig='pub fn new(value: i32) -> Foo' doc=None
method name='fmt' scope=('Foo<Display>',) lang=rust lines=15-17 bytes=193-294 sig='fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result' doc=None
method name='fmt' scope=('Foo<Debug>',) lang=rust lines=21-23 bytes=328-441 sig='fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result' doc=None
trait name='Greet' scope=() lang=rust lines=26-28 bytes=445-495 sig=None doc=None
method name='greet' scope=('Foo<Greet>',) lang=rust lines=31-33 bytes=522-582 sig='fn greet(&self) -> String' doc=None

REFS:
imports raw='std' head='fmt' imported='fmt' for_type=None line=1 col=5
imports raw='crate::helpers' head='assist' imported='assist' for_type=None line=2 col=5
inherits raw='fmt::Display' head=None imported=None for_type='Foo' line=14 col=1
inherits raw='fmt::Debug' head=None imported=None for_type='Foo' line=20 col=1
inherits raw='Greet' head='Greet' imported=None for_type='Foo' line=30 col=1
calls raw='assist' head='assist' imported=None for_type=None line=32 col=9\
""")


class TestNonCodeFixtures:
    def test_markdown_sample(self):
        assert _parse("markdown/sample.md") == snapshot("""\
NODES:
file name='' scope=() lang=markdown lines=1-22 bytes=0-254 sig=None doc=None
section name='Indexter Fixture' scope=() lang=markdown lines=1-22 bytes=0-254 sig='h1' doc=None
section name='Indexter Fixture > Setup' scope=() lang=markdown lines=5-13 bytes=55-157 sig='h2' doc=None
section name='Indexter Fixture > Setup > Prerequisites' scope=() lang=markdown lines=9-13 bytes=97-157 sig='h3' doc=None
section name='Indexter Fixture > Usage' scope=() lang=markdown lines=13-22 bytes=157-254 sig='h2' doc=None
section name='Indexter Fixture > Usage > Basic usage' scope=() lang=markdown lines=15-19 bytes=167-208 sig='h3' doc=None
section name='Indexter Fixture > Usage > Advanced usage' scope=() lang=markdown lines=19-22 bytes=208-254 sig='h3' doc=None

REFS:
""")

    def test_json_sample(self):
        assert _parse("json/sample.json") == snapshot("""\
NODES:
data name='' scope=() lang=json lines=1-8 bytes=0-104 sig='object' doc=None
file name='' scope=() lang=json lines=1-9 bytes=0-105 sig=None doc=None
data name='settings' scope=() lang=json lines=4-7 bytes=52-102 sig='object' doc=None
data name='limits' scope=('settings',) lang=json lines=6-6 bytes=89-98 sig='array' doc=None

REFS:
""")

    def test_yaml_sample(self):
        assert _parse("yaml/sample.yaml") == snapshot("""\
NODES:
data name='' scope=() lang=yaml lines=1-9 bytes=0-84 sig='block_mapping' doc=None
file name='' scope=() lang=yaml lines=1-9 bytes=0-84 sig=None doc=None
data name='settings' scope=() lang=yaml lines=4-9 bytes=36-84 sig='block_mapping' doc=None
data name='limits' scope=('settings',) lang=yaml lines=6-9 bytes=64-84 sig='block_sequence' doc=None

REFS:
""")

    def test_toml_sample(self):
        assert _parse("toml/sample.toml") == snapshot("""\
NODES:
data name='name' scope=() lang=toml lines=1-1 bytes=0-15 sig='pair' doc=None
file name='' scope=() lang=toml lines=1-13 bytes=0-110 sig=None doc=None
data name='version' scope=() lang=toml lines=2-2 bytes=16-27 sig='pair' doc=None
data name='settings' scope=() lang=toml lines=4-8 bytes=29-75 sig='table' doc=None
data name='items' scope=() lang=toml lines=8-11 bytes=75-93 sig='table_array_element' doc=None
data name='items' scope=() lang=toml lines=11-13 bytes=93-110 sig='table_array_element' doc=None

REFS:
""")

    def test_html_sample(self):
        assert _parse("html/sample.html") == snapshot("""\
NODES:
file name='' scope=() lang=html lines=1-17 bytes=0-238 sig=None doc=None
section name='Sample Page' scope=() lang=html lines=7-7 bytes=81-101 sig='h1' doc=None
section name='table' scope=() lang=html lines=8-10 bytes=106-162 sig='table' doc=None
section name='ul-list' scope=() lang=html lines=11-14 bytes=167-219 sig='ul' doc=None

REFS:
""")

    def test_css_sample(self):
        assert _parse("css/sample.css") == snapshot("""\
NODES:
section name='body' scope=() lang=css lines=1-4 bytes=0-48 sig='rule_set' doc=None
file name='' scope=() lang=css lines=1-11 bytes=0-114 sig=None doc=None
section name='@media' scope=() lang=css lines=6-10 bytes=50-113 sig='media_statement' doc=None
section name='body' scope=('@media',) lang=css lines=7-9 bytes=80-111 sig='rule_set' doc=None

REFS:
""")


class TestFallback:
    def test_chunk_fallback_covers_whole_file(self):
        relpath = "other/sample.xyz"
        content = (FIXTURES / relpath).read_text()
        result = ChunkParser(Settings(chunk_size=80, chunk_overlap=10)).parse(relpath, content)
        assert render_result(result) == snapshot("""\
NODES:
chunk name='' scope=() lang= lines=1-2 bytes=0-80 sig=None doc=None
file name='' scope=() lang= lines=1-6 bytes=0-377 sig=None doc=None
chunk name='' scope=() lang= lines=1-3 bytes=70-150 sig=None doc=None
chunk name='' scope=() lang= lines=2-3 bytes=140-220 sig=None doc=None
chunk name='' scope=() lang= lines=3-4 bytes=210-290 sig=None doc=None
chunk name='' scope=() lang= lines=4-5 bytes=280-360 sig=None doc=None
chunk name='' scope=() lang= lines=5-6 bytes=350-377 sig=None doc=None

REFS:
""")


class TestReferenceCountRegression:
    """A plain, non-snapshot assertion: if reference extraction silently
    breaks, this fails with a clear count mismatch instead of a diff that's
    easy to accept without reading."""

    def test_python_fixtures_reference_count(self):
        total_refs = sum(
            len(parse_file(relpath, (FIXTURES / relpath).read_text()).refs)
            for relpath in ("python/sample.py", "python/scopes.py")
        )
        assert total_refs == snapshot(8)
