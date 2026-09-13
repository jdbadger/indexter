"""Selection, roll-up, snippets, budget admission and rendering
(design.md decisions 5, 6, 9, 10).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from indexter.index.resolve import Confidence
from indexter.parse.models import Kind
from indexter.search.types import (
    ContextRef,
    Entry,
    GraphContext,
    MatchReasons,
    Member,
    NodeRow,
    RankedNode,
    Related,
    SearchResponse,
    Selection,
)
from indexter.walk import read_file

# A snippet line longer than this many characters is cut, with a trailing
# marker (decision 6).
SNIPPET_LINE_CUT_CHARS = 240

# Parent kinds that make a node's parent its "container" for roll-up and
# graph context (decisions 5, 7).
CLASS_LIKE_KINDS = frozenset(
    {Kind.CLASS.value, Kind.STRUCT.value, Kind.TRAIT.value, Kind.INTERFACE.value, Kind.ENUM.value}
)


# --- Selection and roll-up ----------------------------------------------------


class _Group:
    __slots__ = ("anchor_id", "anchor_row", "self_reasons", "members", "best")

    def __init__(self, anchor_id: str, anchor_row: NodeRow, best: RankedNode) -> None:
        self.anchor_id = anchor_id
        self.anchor_row = anchor_row
        self.self_reasons: MatchReasons | None = None
        self.members: list[tuple[RankedNode, NodeRow]] = []
        self.best = best


def _container_id(row: NodeRow, node_row: Callable[[str], NodeRow]) -> str | None:
    if row.parent_id is None:
        return None
    parent = node_row(row.parent_id)
    return row.parent_id if parent.kind in CLASS_LIKE_KINDS else None


def _merge_optional_rank(a: int | None, b: int | None) -> int | None:
    if a is None:
        return b
    if b is None:
        return a
    return min(a, b)


def _merge_reasons(a: MatchReasons, b: MatchReasons) -> MatchReasons:
    return MatchReasons(
        vector_rank=_merge_optional_rank(a.vector_rank, b.vector_rank),
        keyword_rank=_merge_optional_rank(a.keyword_rank, b.keyword_rank),
    )


def _finalize(group: _Group) -> Selection:
    all_reasons = [group.self_reasons] if group.self_reasons is not None else []
    all_reasons.extend(ranked_node.reasons for ranked_node, _ in group.members)
    reasons = all_reasons[0]
    for other in all_reasons[1:]:
        reasons = _merge_reasons(reasons, other)

    is_class_entry = (group.self_reasons is not None and len(group.members) >= 1) or len(group.members) >= 2

    if is_class_entry:
        anchor = group.anchor_row
        members = tuple(
            Member(
                node_id=row.id,
                qualified_name=row.qualified_name,
                start_line=row.start_line,
                end_line=row.end_line,
                signature=row.signature,
            )
            for _, row in group.members
        )
        if group.best.node_id == anchor.id:
            snippet_row = anchor
        else:
            snippet_row = next(row for ranked_node, row in group.members if ranked_node is group.best)
        return Selection(
            node_id=anchor.id,
            qualified_name=anchor.qualified_name,
            kind=anchor.kind,
            file_path=anchor.file_path,
            start_line=anchor.start_line,
            end_line=anchor.end_line,
            signature=anchor.signature,
            docstring=anchor.docstring,
            reasons=reasons,
            members=members,
            snippet_node_id=snippet_row.id,
            snippet_start_byte=snippet_row.start_byte,
            snippet_end_byte=snippet_row.end_byte,
        )

    row = group.anchor_row if group.self_reasons is not None else group.members[0][1]
    return Selection(
        node_id=row.id,
        qualified_name=row.qualified_name,
        kind=row.kind,
        file_path=row.file_path,
        start_line=row.start_line,
        end_line=row.end_line,
        signature=row.signature,
        docstring=row.docstring,
        reasons=reasons,
        members=(),
        snippet_node_id=row.id,
        snippet_start_byte=row.start_byte,
        snippet_end_byte=row.end_byte,
    )


def select_entries(
    ranked: Sequence[RankedNode],
    node_row: Callable[[str], NodeRow],
    *,
    limit: int,
) -> tuple[Selection, ...]:
    """Roll fused nodes up into at most `limit` entries (decision 5): a node
    whose parent is class-like joins its container's entry; anything else is
    its own entry. Joining a group never counts against `limit` -- only
    starting one does, so selection keeps walking `ranked` past the naive
    top-`limit` cut to fill slots roll-up frees, and stops for good once
    `limit` entries exist.

    `node_row` is a total function over every ID selection needs: each
    ranked node's own ID, and the parent ID of any node whose `parent_id`
    is not `None`.
    """
    groups: dict[str, _Group] = {}
    order: list[str] = []

    for ranked_node in ranked:
        if len(order) >= limit:
            break
        row = node_row(ranked_node.node_id)
        container_id = _container_id(row, node_row)
        group_key = container_id if container_id is not None else ranked_node.node_id

        group = groups.get(group_key)
        if group is None:
            anchor_row = row if group_key == ranked_node.node_id else node_row(group_key)
            group = _Group(group_key, anchor_row, ranked_node)
            groups[group_key] = group
            order.append(group_key)

        if group_key == ranked_node.node_id:
            group.self_reasons = ranked_node.reasons
        else:
            group.members.append((ranked_node, row))

    return tuple(_finalize(groups[key]) for key in order)


# --- Snippets ------------------------------------------------------------------


def _cut_line(line: str) -> str:
    if len(line) > SNIPPET_LINE_CUT_CHARS:
        return line[:SNIPPET_LINE_CUT_CHARS] + "…"
    return line


def _elide(text: str, *, max_lines: int) -> str:
    lines = [_cut_line(line) for line in text.split("\n")]
    if len(lines) <= max_lines:
        return "\n".join(lines)

    head_count = -(-(max_lines - 1) // 2)  # ceil((n-1)/2)
    tail_count = (max_lines - 1) // 2  # floor((n-1)/2)
    elided_count = len(lines) - head_count - tail_count

    head = lines[:head_count]
    tail = lines[len(lines) - tail_count :] if tail_count else []
    marker = f"… {elided_count} lines elided …"
    return "\n".join([*head, marker, *tail])


def read_snippets(repo: str | Path, selections: Sequence[Selection], *, max_lines: int) -> dict[str, str | None]:
    """Read and middle-elide each selection's snippet, keyed by its entry
    node ID; `None` when the file can't be read or decoded at render time
    (decision 6). Each distinct file is read from disk at most once.
    """
    file_cache: dict[str, str | None] = {}
    snippets: dict[str, str | None] = {}

    for selection in selections:
        if selection.file_path not in file_cache:
            read = read_file(repo, selection.file_path)
            file_cache[selection.file_path] = read[0] if read is not None else None
        content = file_cache[selection.file_path]

        if content is None:
            snippets[selection.node_id] = None
            continue

        try:
            raw = content.encode("utf-8")[selection.snippet_start_byte : selection.snippet_end_byte].decode("utf-8")
        except UnicodeDecodeError:
            snippets[selection.node_id] = None
            continue

        snippets[selection.node_id] = _elide(raw, max_lines=max_lines)

    return snippets


# --- Rendering -----------------------------------------------------------------

# The heading introducing a file group (decision 9).
_RELATED_HEADING = "## related"


def _location(file_path: str, start_line: int, end_line: int) -> str:
    return f"{file_path}:{start_line}-{end_line}"


def _reasons_text(reasons: MatchReasons) -> str:
    parts = []
    if reasons.vector_rank is not None:
        parts.append("vector")
    if reasons.keyword_rank is not None:
        parts.append("keyword")
    return ", ".join(parts)


def _marked_name(ref: ContextRef) -> str:
    """A caller/callee's qualified name, marked when its edge is ambiguous
    (decision 7)."""
    return ref.qualified_name + ("?" if ref.confidence == Confidence.AMBIGUOUS.value else "")


def _shown_of_total(shown: int, total: int) -> str:
    return str(shown) if shown == total else f"{shown} of {total}"


def _context_line(context: GraphContext) -> str | None:
    parts = []
    if context.container is not None:
        parts.append(context.container.qualified_name)
    if context.caller_total:
        shown = ", ".join(_marked_name(c) for c in context.callers)
        count = _shown_of_total(len(context.callers), context.caller_total)
        parts.append(f"callers ({count}): {shown}")
    if context.callee_total:
        shown = ", ".join(_marked_name(c) for c in context.callees)
        count = _shown_of_total(len(context.callees), context.callee_total)
        parts.append(f"callees ({count}): {shown}")
    if not parts:
        return None
    return "in: " + " · ".join(parts)


def _members_block(entry: Entry) -> str:
    lines = ["matched members:"]
    for member in entry.members:
        location = _location(entry.file_path, member.start_line, member.end_line)
        signature = f" — {member.signature}" if member.signature else ""
        lines.append(f"- {member.qualified_name} — {location}{signature}")
    return "\n".join(lines)


def _entry_body(entry: Entry) -> str:
    """One entry's rendered block, without a file heading (decision 9)."""
    location = _location(entry.file_path, entry.start_line, entry.end_line)
    header = f"### {entry.qualified_name} — {entry.kind} — {location}"
    reasons = _reasons_text(entry.reasons)
    if reasons:
        header += f" — {reasons}"

    lines = [header, f"id: {entry.node_id}"]
    context_line = _context_line(entry.context)
    if context_line is not None:
        lines.append(context_line)
    if entry.signature:
        lines.append(entry.signature)
    if entry.docstring:
        lines.append(entry.docstring)
    if entry.members:
        lines.append(_members_block(entry))
    if entry.snippet is not None:
        lines.append(entry.snippet)
    elif entry.snippet_unavailable:
        lines.append("snippet unavailable")
    return "\n".join(lines)


def render_entry_chunk(entry: Entry, *, heading: str | None) -> str:
    """`entry`'s rendered block, prefixed by `heading` (a `## <file>` file
    heading) when it opens a new file group; otherwise the block alone
    (decision 9). Used both to size an entry for budget admission and, for
    the entries that survive, to render the final response -- so a chunk's
    size never differs between the two.
    """
    body = _entry_body(entry)
    return f"{heading}\n\n{body}" if heading is not None else body


def render_related_item(item: Related) -> str:
    location = _location(item.file_path, item.start_line, item.end_line)
    return f"- {item.qualified_name} — {item.kind} — {location} — {item.reason}"


def _header_line(response: SearchResponse) -> str:
    omitted = f" ({response.entries_omitted} omitted for budget)" if response.entries_omitted else ""
    return f'{len(response.entries)} results for "{response.query}"{omitted}'


# --- Budget admission ------------------------------------------------------------


def admit_entries(chunks: Sequence[str], *, budget: int) -> int:
    """How many of `chunks` (each entry's rendered block, in rank order,
    from `render_entry_chunk`) fit within `budget` characters (decision 9):
    the first is always admitted, and admission stops -- without pulling
    later, smaller chunks forward -- at the first one that would not fit.
    Returns the admitted count; the caller slices its own parallel entry
    list by it.
    """
    admitted: list[str] = []
    for chunk in chunks:
        candidate = "\n\n".join([*admitted, chunk])
        if not admitted or len(candidate) <= budget:
            admitted.append(chunk)
        else:
            break
    return len(admitted)


def admit_related(entries_text: str, related_chunks: Sequence[str], *, budget: int) -> int:
    """How many of `related_chunks` (each from `render_related_item`, in
    score order) fit in what remains of `budget` after `entries_text` (the
    joined, already-admitted entry chunks) (decision 9). Unlike entries,
    there is no "always admit the first related item" rule -- `related` may
    end up empty. Returns the admitted count.
    """
    if not related_chunks:
        return 0
    admitted: list[str] = []
    for chunk in related_chunks:
        section = "\n".join([_RELATED_HEADING, *admitted, chunk])
        candidate = f"{entries_text}\n\n{section}" if entries_text else section
        if len(candidate) <= budget:
            admitted.append(chunk)
        else:
            break
    return len(admitted)


# --- Final rendering ---------------------------------------------------------


def render(response: SearchResponse) -> str:
    """The full deterministic plain-text rendering of `response` (decision
    9): a header line, entries grouped by file (files ordered by their best
    entry's rank, entries within a file in rank order), then a `related`
    section. A pure function of `response` alone, so the same response
    always renders identically.
    """
    if not response.entries:
        return f'no results for "{response.query}"'

    file_order: list[str] = []
    by_file: dict[str, list[Entry]] = {}
    seen_files: set[str] = set()
    chunk_by_node_id: dict[str, str] = {}
    for entry in response.entries:
        heading = f"## {entry.file_path}" if entry.file_path not in seen_files else None
        seen_files.add(entry.file_path)
        chunk_by_node_id[entry.node_id] = render_entry_chunk(entry, heading=heading)
        if entry.file_path not in by_file:
            by_file[entry.file_path] = []
            file_order.append(entry.file_path)
        by_file[entry.file_path].append(entry)

    ordered_chunks = [chunk_by_node_id[entry.node_id] for file_path in file_order for entry in by_file[file_path]]
    entries_text = "\n\n".join(ordered_chunks)
    text = f"{_header_line(response)}\n\n{entries_text}"

    if response.related:
        related_section = "\n".join([_RELATED_HEADING, *(render_related_item(item) for item in response.related)])
        text += f"\n\n{related_section}"

    return text
