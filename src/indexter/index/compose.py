"""Compose each parsed node into embeddable text -- see design.md decision 6.

`compose_file` turns one file's `ParseResult` into, per node ID, the text an
embedder will see (`embed_text`), a hash of that text for change detection
(`embed_hash`), a dotted `qualified_name`, whitespace-split `name_words`, and
the FTS `body` (decision 11: a node's own byte range minus its children's).

Sections are joined most-meaningful-first so truncation to the model's token
budget (decision 7) drops the least useful text first: label, signature (or
a declaration-line fallback), docstring prose, a per-kind body variant, then
structured documentation (Args/Returns/Raises, JSDoc tags, rustdoc
sections) last.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import TYPE_CHECKING

from indexter.parse.models import Kind, ParsedNode, ParseResult

if TYPE_CHECKING:
    from indexter.index.embed import TokenizerLike

INDEX_FORMAT_VERSION = 1

_CONTAINER_KINDS = frozenset({Kind.CLASS, Kind.STRUCT, Kind.TRAIT, Kind.INTERFACE, Kind.ENUM})
_LEAF_CODE_KINDS = frozenset({Kind.FUNCTION, Kind.METHOD, Kind.CONSTANT, Kind.TYPE_ALIAS})

_KIND_PLURAL = {
    Kind.FILE: "files",
    Kind.CLASS: "classes",
    Kind.FUNCTION: "functions",
    Kind.METHOD: "methods",
    Kind.CONSTANT: "constants",
    Kind.INTERFACE: "interfaces",
    Kind.TYPE_ALIAS: "type aliases",
    Kind.ENUM: "enums",
    Kind.STRUCT: "structs",
    Kind.TRAIT: "traits",
    Kind.SECTION: "sections",
    Kind.DATA: "data",
    Kind.CHUNK: "chunks",
    Kind.EXTERNAL_MODULE: "external modules",
}

_STRUCTURED_HEADERS = frozenset(
    {
        "args",
        "arguments",
        "parameters",
        "params",
        "returns",
        "return",
        "yields",
        "yield",
        "raises",
        "raise",
        "throws",
        "attributes",
        "examples",
        "example",
        "notes",
        "note",
        "errors",
        "panics",
        "safety",
    }
)

_JSDOC_TAG_RE = re.compile(r"^@(param|returns?|throws?|arg|argument|yields?|example)\b", re.IGNORECASE)
_MD_HEADER_RE = re.compile(r"^#{1,6}\s+(.+)$")
_WORD_SEPARATORS = re.compile(r"[_\-\s]+")


@dataclass(frozen=True, slots=True)
class ComposedNode:
    """One node's composed output. See module docstring for the fields."""

    embed_text: str
    embed_hash: str
    qualified_name: str
    name_words: str
    body: str


# --- Identifier and path splitting ------------------------------------------


def split_identifier(name: str) -> list[str]:
    """Split an identifier into lowercase words: `_`/`-`/whitespace as
    boundaries, then camelCase/PascalCase/acronym runs and digit runs.
    `get_user_by_email` -> `["get", "user", "by", "email"]`;
    `HTTPServer2` -> `["http", "server", "2"]`.
    """
    if not name:
        return []
    words: list[str] = []
    for chunk in _WORD_SEPARATORS.split(name):
        if chunk:
            words.extend(_split_camel_chunk(chunk))
    return [w.lower() for w in words if w]


def _split_camel_chunk(chunk: str) -> list[str]:
    words: list[str] = []
    i, n = 0, len(chunk)
    while i < n:
        c = chunk[i]
        if c.isdigit():
            j = i + 1
            while j < n and chunk[j].isdigit():
                j += 1
        elif c.isupper():
            j = i + 1
            while j < n and chunk[j].isupper():
                j += 1
            if j - i > 1 and j < n and chunk[j].islower():
                j -= 1  # acronym run followed by a capitalized word: back off one
            elif j - i == 1 and j < n and chunk[j].islower():
                k = j
                while k < n and chunk[k].islower():
                    k += 1
                j = k
        else:
            j = i + 1
            while j < n and chunk[j].islower():
                j += 1
        words.append(chunk[i:j])
        i = j
    return words


def split_path_words(relpath: str) -> list[str]:
    """A relative path's words, dropping the final extension:
    `src/auth/handlers.py` -> `["src", "auth", "handlers"]`.
    """
    parts = list(PurePosixPath(relpath).parts)
    if parts:
        parts[-1] = PurePosixPath(parts[-1]).stem
    words: list[str] = []
    for part in parts:
        words.extend(split_identifier(part))
    return words


# --- Docstring splitting -----------------------------------------------------


def split_docstring(docstring: str) -> tuple[str, str]:
    """Split a docstring into prose and a trailing structured-documentation
    block, recognizing Google/NumPy-style headers (`Args`, `Returns`, ...),
    JSDoc tags (`@param`, ...), and rustdoc `# Arguments`/`# Errors`/
    `# Examples` sections. Everything from the first recognized header
    onward is "structured"; a docstring with none is entirely prose.
    """
    if not docstring:
        return "", ""
    lines = docstring.splitlines()
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            continue
        if _is_structured_header(stripped):
            prose = "\n".join(lines[:i]).strip()
            structured = "\n".join(lines[i:]).strip()
            return prose, structured
    return docstring.strip(), ""


def _is_structured_header(stripped: str) -> bool:
    header = stripped.rstrip(":").strip().lower()
    if header in _STRUCTURED_HEADERS:
        return True
    if _JSDOC_TAG_RE.match(stripped):
        return True
    md_match = _MD_HEADER_RE.match(stripped)
    return bool(md_match and md_match.group(1).strip().lower() in _STRUCTURED_HEADERS)


# --- Composition --------------------------------------------------------------


def _qualified_name(node: ParsedNode, relpath: str) -> str:
    if node.kind == Kind.FILE:
        return relpath
    parts = [*node.scope_path, node.name] if node.name else list(node.scope_path)
    return ".".join(parts) if parts else relpath


def _name_words(node: ParsedNode, path_words: list[str]) -> str:
    if node.kind == Kind.FILE:
        return " ".join(path_words)
    parts = [*node.scope_path, node.name] if node.name else list(node.scope_path)
    words: list[str] = []
    for part in parts:
        words.extend(split_identifier(part))
    return " ".join(words)


def _header_line(node: ParsedNode, raw: str) -> str:
    if node.signature is not None:
        return node.signature
    return raw.splitlines()[0].strip() if raw else ""


def _label_line(node: ParsedNode, qualified_name: str, name_words: str, relpath: str, path_words: list[str]) -> str:
    return f"{node.kind.value} {qualified_name} | {name_words} | {relpath} ({' '.join(path_words)})"


def _residue_ranges(node: ParsedNode, children: list[ParsedNode]) -> list[tuple[int, int]]:
    if not children:
        return [(node.start_byte, node.end_byte)]
    intervals = sorted((c.start_byte, c.end_byte) for c in children)
    merged: list[list[int]] = []
    for s, e in intervals:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    gaps: list[tuple[int, int]] = []
    cursor = node.start_byte
    for s, e in merged:
        if s > cursor:
            gaps.append((cursor, s))
        cursor = max(cursor, e)
    if cursor < node.end_byte:
        gaps.append((cursor, node.end_byte))
    return gaps


def _residue_text(node: ParsedNode, children: list[ParsedNode], source_bytes: bytes) -> str:
    ranges = _residue_ranges(node, children)
    return "".join(source_bytes[s:e].decode("utf-8", errors="replace") for s, e in ranges)


def _member_listing(children: list[ParsedNode]) -> str:
    if not children:
        return ""
    grouped: dict[Kind, list[str]] = {}
    for child in children:
        grouped.setdefault(child.kind, []).append(child.name or "(anonymous)")
    lines = [f"{_KIND_PLURAL.get(kind, f'{kind.value}s')}: {', '.join(names)}" for kind, names in grouped.items()]
    return "\n".join(lines)


def _collapse_blank_lines(text: str) -> str:
    out: list[str] = []
    blank = False
    for line in text.splitlines():
        if line.strip() == "":
            if not blank:
                out.append("")
            blank = True
        else:
            out.append(line)
            blank = False
    return "\n".join(out)


def _body_prefix(residue: str, header: str, docstring: str | None) -> str:
    text = residue
    if header and text.startswith(header):
        text = text[len(header) :]
        if text.startswith(":"):  # a signature's trailing colon is stripped from `header` itself
            text = text[1:]
    if docstring:
        idx = text.find(docstring)
        if idx != -1:
            end = idx + len(docstring)
            # `docstring` had its surrounding whitespace stripped (see
            # `_strip_docstring`), but the source between its last real
            # character and the closing quote(s) still has it.
            scan = end
            while scan < len(text) and text[scan] in " \t\n":
                scan += 1
            for quote in ('"""', "'''", '"', "'"):
                if text[scan : scan + len(quote)] == quote:
                    end = scan + len(quote)
                    break
            text = text[end:]
    return _collapse_blank_lines(text).strip()


def _section_four(node: ParsedNode, children: list[ParsedNode], residue: str, header: str) -> str:
    if node.kind in _CONTAINER_KINDS:
        return _member_listing(children)
    if node.kind == Kind.FILE:
        listing = _member_listing(children)
        resid = residue.strip()
        if listing and resid:
            return f"{listing}\n\n{resid}"
        return listing or resid
    if node.kind in _LEAF_CODE_KINDS:
        return _body_prefix(residue, header, node.docstring)
    return residue.strip()


def _truncate(text: str, tokenizer: TokenizerLike, budget: int) -> str:
    """Cut `text` to at most `budget - 2` tokens (room for `[CLS]`/`[SEP]`),
    at the character offset where the last in-budget token ends.
    """
    if not text:
        return text
    limit = max(budget - 2, 0)
    encoding = tokenizer.encode(text, add_special_tokens=False)
    ids = encoding.ids
    if len(ids) <= limit:
        return text
    if limit == 0:
        return ""
    end_offset = encoding.offsets[limit - 1][1]
    return text[:end_offset]


def _compose_chunk(node: ParsedNode, relpath: str, path_words: list[str], source_bytes: bytes) -> tuple[str, str, str]:
    qualified_name = f"{relpath}:{node.start_line}-{node.end_line}"
    name_words = " ".join(path_words)
    raw = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
    embed_text = f"{qualified_name}\n{raw}"
    return qualified_name, name_words, embed_text


def _compose_symbol(
    node: ParsedNode,
    relpath: str,
    path_words: list[str],
    children: list[ParsedNode],
    residue: str,
    raw: str,
) -> tuple[str, str, str]:
    qualified_name = _qualified_name(node, relpath)
    name_words = _name_words(node, path_words)
    header = _header_line(node, raw)

    sections = [_label_line(node, qualified_name, name_words, relpath, path_words)]
    if header:
        sections.append(header)
    prose, structured = split_docstring(node.docstring) if node.docstring else ("", "")
    if prose:
        sections.append(prose)
    body_section = _section_four(node, children, residue, header)
    if body_section.strip():
        sections.append(body_section.strip())
    if structured:
        sections.append(structured)

    return qualified_name, name_words, "\n".join(sections)


def compose_file(
    relpath: str,
    content: str,
    parse_result: ParseResult,
    tokenizer: TokenizerLike,
    budget: int,
) -> dict[str, ComposedNode]:
    """Compose every node `parse_result` produced for one file's `content`.

    `budget` is the model's max token count (`Settings.embed_max_tokens`);
    truncation reserves two tokens for `[CLS]`/`[SEP]`.
    """
    source_bytes = content.encode("utf-8")
    nodes = parse_result.nodes

    children_by_parent: dict[str, list[ParsedNode]] = {}
    for node in nodes:
        if node.parent_id is not None:
            children_by_parent.setdefault(node.parent_id, []).append(node)
    for kids in children_by_parent.values():
        kids.sort(key=lambda n: n.start_byte)

    path_words = split_path_words(relpath)
    result: dict[str, ComposedNode] = {}

    for node in nodes:
        children = children_by_parent.get(node.id, [])
        residue = _residue_text(node, children, source_bytes)

        if node.kind == Kind.CHUNK:
            qualified_name, name_words, embed_text = _compose_chunk(node, relpath, path_words, source_bytes)
        else:
            raw = source_bytes[node.start_byte : node.end_byte].decode("utf-8", errors="replace")
            qualified_name, name_words, embed_text = _compose_symbol(node, relpath, path_words, children, residue, raw)

        truncated = _truncate(embed_text, tokenizer, budget)
        result[node.id] = ComposedNode(
            embed_text=truncated,
            embed_hash=hashlib.sha256(truncated.encode()).hexdigest(),
            qualified_name=qualified_name,
            name_words=name_words,
            body=residue.strip(),
        )

    return result
