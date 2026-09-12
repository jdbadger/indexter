"""Deterministic text rendering of a `ParseResult`, shared by every snapshot
test in this package.

`id`/`parent_id` are left out. They're derived entirely from fields already
rendered here (`scope_path`, `name`, `kind`, plus the relpath each test
passes in) via the file-wide duplicate-suffix and parent-linking passes that
`test_ids.py` already covers directly -- including them would just repeat
that coverage in every language's snapshot and force every one of them to
churn if the ID format itself ever changes, for no signal specific to the
parser under test.

Both nodes and refs are sorted before rendering so the snapshot doesn't
depend on tree-sitter's match iteration order, which the query compiler
gives no ordering guarantee over.
"""

from __future__ import annotations

from indexter.parse.models import ParseResult


def render_result(result: ParseResult) -> str:
    node_lines = [
        f"{n.kind.value} name={n.name!r} scope={n.scope_path} lang={n.language} "
        f"lines={n.start_line}-{n.end_line} bytes={n.start_byte}-{n.end_byte} "
        f"sig={n.signature!r} doc={n.docstring!r}"
        for n in sorted(result.nodes, key=lambda n: (n.start_byte, n.end_byte, n.kind.value, n.name))
    ]
    ref_lines = [
        f"{r.ref_kind.value} raw={r.raw_name!r} head={r.head!r} line={r.line} col={r.col}"
        for r in sorted(result.refs, key=lambda r: (r.line, r.col, r.raw_name))
    ]
    return "NODES:\n" + "\n".join(node_lines) + "\n\nREFS:\n" + "\n".join(ref_lines)
