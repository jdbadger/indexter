"""Node identity: deterministic IDs, duplicate disambiguation, parent
linking, and reference-to-node linking.

These three passes run over one file's nodes as a unit, after every node in
the file has been collected -- individual language parsers never assign IDs
themselves (see design.md decision 4). Parent and reference linking both
work by byte-range containment rather than by re-walking scope names, which
means they get decorated/exported wrapping right for free: whatever byte
range a parser chose for a node's `ParsedNode` is exactly what containment
checks against.
"""

from __future__ import annotations

import dataclasses

from indexter.parse.models import Kind, ParsedNode, ParsedRef, RawRef


def build_id(relpath: str, scope_path: tuple[str, ...], name: str, kind: Kind) -> str:
    """`<relpath>::<scope>.<name>#<kind>`, omitting the scope segment at file
    scope and the whole qualified segment when there is no name either (the
    file node itself): `<relpath>::#file`.
    """
    parts = [*scope_path]
    if name:
        parts.append(name)
    qualified = ".".join(parts)
    if qualified:
        return f"{relpath}::{qualified}#{kind.value}"
    return f"{relpath}::#{kind.value}"


def assign_ids(relpath: str, nodes: list[ParsedNode]) -> list[ParsedNode]:
    """Assign each node its ID, appending `~2`, `~3`, ... to genuine
    duplicates (same relpath, scope, name, and kind) in ascending start-line
    order, leaving the first occurrence unsuffixed.
    """
    base_ids = [build_id(relpath, n.scope_path, n.name, n.kind) for n in nodes]

    groups: dict[str, list[int]] = {}
    for i, base_id in enumerate(base_ids):
        groups.setdefault(base_id, []).append(i)

    final_ids: list[str] = [""] * len(nodes)
    for base_id, indices in groups.items():
        ordered = sorted(indices, key=lambda i: nodes[i].start_line)
        for rank, i in enumerate(ordered):
            final_ids[i] = base_id if rank == 0 else f"{base_id}~{rank + 1}"

    return [dataclasses.replace(n, id=final_ids[i]) for i, n in enumerate(nodes)]


def _is_proper_ancestor(candidate: ParsedNode, node: ParsedNode) -> bool:
    if (candidate.start_byte, candidate.end_byte) == (node.start_byte, node.end_byte):
        return False
    return candidate.start_byte <= node.start_byte and node.end_byte <= candidate.end_byte


def link_parents(nodes: list[ParsedNode]) -> list[ParsedNode]:
    """Link each node to the innermost node (by byte range) that properly
    contains it. Nodes are assumed to already carry their final `id`.
    """
    result = []
    for node in nodes:
        ancestors = [other for other in nodes if _is_proper_ancestor(other, node)]
        parent = min(ancestors, key=lambda a: a.end_byte - a.start_byte, default=None)
        result.append(dataclasses.replace(node, parent_id=parent.id if parent else None))
    return result


def _contains_point(node: ParsedNode, point: int) -> bool:
    if node.start_byte == node.end_byte:
        return node.start_byte == point
    return node.start_byte <= point < node.end_byte


def link_refs(nodes: list[ParsedNode], raw_refs: list[RawRef]) -> list[ParsedRef]:
    """Resolve each `RawRef.origin_byte` to the innermost node (by byte
    range) containing it, producing the finished `ParsedRef` list.
    """
    refs = []
    for raw in raw_refs:
        containing = [n for n in nodes if _contains_point(n, raw.origin_byte)]
        origin = min(containing, key=lambda n: n.end_byte - n.start_byte, default=None)
        refs.append(
            ParsedRef(
                from_node_id=origin.id if origin else "",
                raw_name=raw.raw_name,
                head=raw.head,
                ref_kind=raw.ref_kind,
                line=raw.line,
                col=raw.col,
                imported_name=raw.imported_name,
                for_type=raw.for_type,
            )
        )
    return refs
