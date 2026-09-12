"""Module and member resolution: the machinery group 5's tiers are built on.

Resolution runs whole-repo, in memory, once per run (design.md decision 1),
so this module works over a `RepoIndex` holding every node and ref rather
than querying the database per lookup. `RepoIndex` is built once (either
loaded from a database via `load_repo_index`, or assembled directly in
tests) and every function below is a pure lookup against it.

Two kinds of resolution live here:

- **Module resolution** (`resolve_python_module`, `resolve_js_specifier`,
  `resolve_rust_path`) turns an import's module part into a repository file
  or an external package -- `ModuleResolution`.
- **Member lookup** (`resolve_python_member`, `resolve_js_member`,
  `resolve_rust_member`, `class_member`) turns a name into a specific node
  within an already-resolved module or class -- `Target`.

Tiers themselves (which lookup applies to which reference, in what order)
are group 5's job; this module only supplies the primitives they compose.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, replace
from enum import StrEnum

from indexter.parse.models import Kind, RefKind

LANGUAGE_FAMILIES: dict[str, str] = {
    "python": "python",
    "javascript": "javascript",
    "typescript": "javascript",
    "rust": "rust",
}


def family_of(language: str | None) -> str | None:
    """The resolution language family for a node/ref's `language`, or
    `None` for languages resolution doesn't cover (CSS, HTML, ...).
    References never resolve across families (design.md decision 5)."""
    return LANGUAGE_FAMILIES.get(language) if language else None


@dataclass(frozen=True, slots=True)
class ResolveNode:
    """One `nodes` row's resolution-relevant fields."""

    id: str
    kind: Kind
    name: str
    qualified_name: str
    file_path: str
    language: str | None
    parent_id: str | None


@dataclass(frozen=True, slots=True)
class ResolveRef:
    """One `refs` row's resolution-relevant fields."""

    id: int
    from_node_id: str
    raw_name: str
    head: str | None
    imported_name: str | None
    for_type: str | None
    ref_kind: RefKind
    line: int | None
    col: int | None


class RepoIndex:
    """Every node and ref in the repository, indexed for the lookups module
    and member resolution need. Built once per resolution run."""

    def __init__(self, nodes: list[ResolveNode], refs: list[ResolveRef]) -> None:
        self.nodes = nodes
        self.refs = refs
        self._by_id: dict[str, ResolveNode] = {n.id: n for n in nodes}
        self._file_node_by_path: dict[str, ResolveNode] = {}
        self._children: dict[str, list[ResolveNode]] = {}
        self._nodes_by_file: dict[str, list[ResolveNode]] = {}
        self._by_name: dict[str, list[ResolveNode]] = {}
        for n in nodes:
            if n.kind == Kind.FILE:
                self._file_node_by_path[n.file_path] = n
            if n.parent_id is not None:
                self._children.setdefault(n.parent_id, []).append(n)
            self._nodes_by_file.setdefault(n.file_path, []).append(n)
            self._by_name.setdefault(n.name, []).append(n)

        # An import binding is "visible" from an origin node or any of its
        # ancestors (group 5 tier 3, design.md decision 5): indexed by the
        # exact node it originates from, keyed by the name it binds. First
        # one wins per (origin, head); good enough for the rare case of two
        # imports at the same origin binding the same name. A file-scope
        # import ref (origin is the file node itself) is also "the module's
        # own binding" for re-export following (decision 6); no separate
        # index is needed for that -- it's the same dict keyed by a file
        # node's ID rather than a function's.
        self._imports_by_origin: dict[str, dict[str, ResolveRef]] = {}
        self._wildcard_imports_by_origin: dict[str, list[ResolveRef]] = {}
        self._imports_in_file: dict[str, list[ResolveRef]] = {}
        for r in sorted(refs, key=lambda r: r.id):
            if r.ref_kind != RefKind.IMPORTS:
                continue
            origin = self._by_id.get(r.from_node_id)
            if origin is None:
                continue
            self._imports_in_file.setdefault(origin.file_path, []).append(r)
            if r.imported_name == "*":
                self._wildcard_imports_by_origin.setdefault(r.from_node_id, []).append(r)
            elif r.head is not None:
                self._imports_by_origin.setdefault(r.from_node_id, {}).setdefault(r.head, r)

    def node(self, node_id: str) -> ResolveNode | None:
        return self._by_id.get(node_id)

    def has_file(self, path: str) -> bool:
        return path in self._file_node_by_path

    def file_node(self, path: str) -> ResolveNode | None:
        return self._file_node_by_path.get(path)

    def file_node_id(self, path: str) -> str | None:
        node = self._file_node_by_path.get(path)
        return node.id if node is not None else None

    def children(self, node_id: str) -> list[ResolveNode]:
        return self._children.get(node_id, [])

    def top_level(self, path: str) -> list[ResolveNode]:
        file_node = self.file_node(path)
        return self.children(file_node.id) if file_node is not None else []

    def top_level_named(self, path: str, name: str) -> ResolveNode | None:
        matches = [n for n in self.top_level(path) if n.name == name]
        return min(matches, key=lambda n: n.id) if matches else None

    def file_level_import(self, path: str, head: str) -> ResolveRef | None:
        """The file-scope import ref whose `head` binds `head` in this
        file's own namespace -- used to follow re-exports."""
        file_node = self.file_node(path)
        return self._imports_by_origin.get(file_node.id, {}).get(head) if file_node is not None else None

    def visible_import(self, origin_id: str, head: str) -> ResolveRef | None:
        """The import binding for `head` visible from `origin_id`: the
        innermost of the origin and its ancestors that binds it (design.md
        decision 5 tier 3 -- a function-local import shadows a file-level
        one)."""
        node: ResolveNode | None = self._by_id.get(origin_id)
        while node is not None:
            ref = self._imports_by_origin.get(node.id, {}).get(head)
            if ref is not None:
                return ref
            node = self._by_id.get(node.parent_id) if node.parent_id else None
        return None

    def visible_wildcards(self, origin_id: str) -> list[ResolveRef]:
        """Wildcard imports (`from a import *`, `use a::*`) visible from
        `origin_id`, from the innermost scope that has any."""
        node: ResolveNode | None = self._by_id.get(origin_id)
        while node is not None:
            refs = self._wildcard_imports_by_origin.get(node.id)
            if refs:
                return sorted(refs, key=lambda r: r.id)
            node = self._by_id.get(node.parent_id) if node.parent_id else None
        return []

    def imports_in_file(self, path: str) -> list[ResolveRef]:
        return self._imports_in_file.get(path, [])

    def nodes_by_file(self, path: str) -> list[ResolveNode]:
        return self._nodes_by_file.get(path, [])

    def by_name(self, name: str) -> list[ResolveNode]:
        return self._by_name.get(name, [])

    def rust_type_members(self, type_name: str) -> list[ResolveNode]:
        """Every method, in any Rust file, whose scope path is `type_name`
        with or without a `<Trait>` suffix (design.md decision 4) -- Rust
        methods are never children of their type's node."""
        results = []
        for n in self.nodes:
            if n.kind != Kind.METHOD or family_of(n.language) != "rust":
                continue
            scope, _, _member = n.qualified_name.rpartition(".")
            if not scope:
                continue
            if scope == type_name or (scope.startswith(f"{type_name}<") and scope.endswith(">")):
                results.append(n)
        return sorted(results, key=lambda n: n.id)

    def python_suffix_matches(self, components: list[str]) -> list[str]:
        """Repository Python files whose module path ends, at a
        path-component boundary, with `components` (design.md decision 6)."""
        matches = []
        for path, file_node in self._file_node_by_path.items():
            if family_of(file_node.language) != "python":
                continue
            suffix = _python_module_components(path)
            if len(suffix) >= len(components) and suffix[len(suffix) - len(components) :] == components:
                matches.append(path)
        return matches


def load_repo_index(conn: sqlite3.Connection) -> RepoIndex:
    """Load every node and ref into a `RepoIndex` for one resolution run."""
    node_rows = conn.execute(
        "SELECT id, kind, name, qualified_name, file_path, language, parent_id FROM nodes"
    ).fetchall()
    nodes = [
        ResolveNode(
            id=row["id"],
            kind=Kind(row["kind"]),
            name=row["name"],
            qualified_name=row["qualified_name"] or "",
            file_path=row["file_path"],
            language=row["language"],
            parent_id=row["parent_id"],
        )
        for row in node_rows
    ]
    ref_rows = conn.execute(
        "SELECT id, from_node_id, raw_name, head, imported_name, for_type, ref_kind, line, col "
        "FROM refs ORDER BY id"
    ).fetchall()
    refs = [
        ResolveRef(
            id=row["id"],
            from_node_id=row["from_node_id"],
            raw_name=row["raw_name"],
            head=row["head"],
            imported_name=row["imported_name"],
            for_type=row["for_type"],
            ref_kind=RefKind(row["ref_kind"]),
            line=row["line"],
            col=row["col"],
        )
        for row in ref_rows
    ]
    return RepoIndex(nodes, refs)


@dataclass(frozen=True, slots=True)
class ModuleResolution:
    """Where an import's module part landed, before its member is looked
    up: a repository file, an external package, or neither (a relative
    import, or a Rust path deep segment, matching no repository file)."""

    file_path: str | None = None
    external_name: str | None = None


@dataclass(frozen=True, slots=True)
class Target:
    """What a name chain finally landed on: a node, an external module, or
    nothing (`failed`, in group 5's vocabulary)."""

    node_id: str | None = None
    external_name: str | None = None


# --- Python -------------------------------------------------------------


def _python_module_components(path: str) -> list[str]:
    if path.endswith("/__init__.py"):
        return path[: -len("/__init__.py")].split("/")
    return path.removesuffix(".py").split("/")


def _python_package_dir(module_file: str) -> str:
    if module_file.endswith("/__init__.py"):
        return module_file[: -len("/__init__.py")]
    if module_file == "__init__.py":
        return ""
    return module_file.removesuffix(".py")


def _shared_prefix_len(file_path: str, importer_dir: str) -> int:
    file_dir = file_path.rsplit("/", 1)[0] if "/" in file_path else ""
    file_parts = file_dir.split("/") if file_dir else []
    importer_parts = importer_dir.split("/") if importer_dir else []
    n = 0
    for a, b in zip(file_parts, importer_parts, strict=False):
        if a != b:
            break
        n += 1
    return n


def _resolve_python_relative(raw_name: str, importer_path: str, index: RepoIndex) -> ModuleResolution:
    level = len(raw_name) - len(raw_name.lstrip("."))
    rest = raw_name[level:]
    importer_dir = importer_path.rsplit("/", 1)[0] if "/" in importer_path else ""
    base_parts = importer_dir.split("/") if importer_dir else []
    up = level - 1
    if up > len(base_parts):
        return ModuleResolution()
    target_parts = base_parts[: len(base_parts) - up] if up else base_parts
    if rest:
        target_parts = [*target_parts, *rest.split(".")]
    prefix = "/".join(target_parts)
    module_py = f"{prefix}.py" if prefix else None
    package_init = f"{prefix}/__init__.py" if prefix else "__init__.py"
    if module_py and index.has_file(module_py):
        return ModuleResolution(file_path=module_py)
    if index.has_file(package_init):
        return ModuleResolution(file_path=package_init)
    return ModuleResolution()


def resolve_python_module(raw_name: str, importer_path: str, index: RepoIndex) -> ModuleResolution:
    """Relative modules resolve against the importer's package directory;
    absolute dotted modules match repository files by suffix, tie-broken by
    longest shared directory prefix with the importer then shortest path;
    an absolute module matching nothing is external, named by its first
    segment (design.md decision 6)."""
    if raw_name.startswith("."):
        return _resolve_python_relative(raw_name, importer_path, index)

    components = [c for c in raw_name.split(".") if c]
    if not components:
        return ModuleResolution()

    candidates = index.python_suffix_matches(components)
    if not candidates:
        return ModuleResolution(external_name=components[0])

    importer_dir = importer_path.rsplit("/", 1)[0] if "/" in importer_path else ""
    chosen = min(candidates, key=lambda fp: (-_shared_prefix_len(fp, importer_dir), len(fp), fp))
    return ModuleResolution(file_path=chosen)


def resolve_python_member(module_file: str, name: str, index: RepoIndex) -> Target:
    """`m.n`: `n` tried first as a submodule of `m`, then as a top-level
    node in `m`, then as a name `m` itself imports (a re-export), following
    at most 5 hops, cycle-safe. When `m` resolves but `n` doesn't anywhere
    in the chain, the target is `m`'s own file node (design.md decision 6)."""
    current = module_file
    current_name = name
    visited: set[str] = set()
    for _ in range(6):  # the module itself, plus up to 5 re-export hops
        if current in visited:
            break
        visited.add(current)

        pkg_dir = _python_package_dir(current)
        sub_py = f"{pkg_dir}/{current_name}.py" if pkg_dir else f"{current_name}.py"
        sub_pkg = f"{pkg_dir}/{current_name}/__init__.py" if pkg_dir else f"{current_name}/__init__.py"
        if index.has_file(sub_py):
            return Target(node_id=index.file_node_id(sub_py))
        if index.has_file(sub_pkg):
            return Target(node_id=index.file_node_id(sub_pkg))

        top_level = index.top_level_named(current, current_name)
        if top_level is not None:
            return Target(node_id=top_level.id)

        rebind = index.file_level_import(current, current_name)
        if rebind is None or not rebind.imported_name or rebind.imported_name == "*":
            break
        next_module = resolve_python_module(rebind.raw_name, current, index)
        if next_module.external_name is not None:
            return Target(external_name=next_module.external_name)
        if next_module.file_path is None:
            break
        current = next_module.file_path
        current_name = rebind.imported_name

    file_node = index.file_node(module_file)
    return Target(node_id=file_node.id) if file_node is not None else Target()


# --- JavaScript / TypeScript ----------------------------------------------

_JS_EXT_ATTEMPTS = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")
_JS_TO_TS_EXT = {".js": ".ts", ".jsx": ".tsx", ".mjs": ".mts", ".cjs": ".cts"}


def _normalize_relative(base_dir: str, specifier: str) -> str:
    parts = base_dir.split("/") if base_dir else []
    for seg in specifier.split("/"):
        if seg in ("", "."):
            continue
        if seg == "..":
            if parts:
                parts.pop()
            continue
        parts.append(seg)
    return "/".join(parts)


def resolve_js_specifier(specifier: str, importer_path: str, index: RepoIndex) -> ModuleResolution:
    """Relative specifiers resolve against the importer's directory, trying
    the exact path, extensions appended, an `index` file, and (for a
    `.js`-family specifier) the TypeScript counterpart. Anything else is
    external, named by its package (design.md decision 6)."""
    if specifier.startswith("./") or specifier.startswith("../"):
        importer_dir = importer_path.rsplit("/", 1)[0] if "/" in importer_path else ""
        candidate = _normalize_relative(importer_dir, specifier)

        if index.has_file(candidate):
            return ModuleResolution(file_path=candidate)
        for ext in _JS_EXT_ATTEMPTS:
            if index.has_file(candidate + ext):
                return ModuleResolution(file_path=candidate + ext)
        for ext in _JS_EXT_ATTEMPTS:
            idx = f"{candidate}/index{ext}"
            if index.has_file(idx):
                return ModuleResolution(file_path=idx)
        for js_ext, ts_ext in _JS_TO_TS_EXT.items():
            if candidate.endswith(js_ext):
                ts_candidate = candidate[: -len(js_ext)] + ts_ext
                if index.has_file(ts_candidate):
                    return ModuleResolution(file_path=ts_candidate)
        return ModuleResolution()

    if specifier.startswith("@"):
        segments = specifier.split("/")
        name = "/".join(segments[:2]) if len(segments) >= 2 else specifier
    else:
        name = specifier.split("/")[0]
    return ModuleResolution(external_name=name)


def _js_default_target(module_file: str, binding: str | None, index: RepoIndex) -> Target:
    if binding:
        top = index.top_level_named(module_file, binding)
        if top is not None:
            return Target(node_id=top.id)
    candidates = [n for n in index.top_level(module_file) if n.kind in (Kind.CLASS, Kind.FUNCTION)]
    if len(candidates) == 1:
        return Target(node_id=candidates[0].id)
    file_node = index.file_node(module_file)
    return Target(node_id=file_node.id) if file_node is not None else Target()


def resolve_js_member(module_file: str, name: str, index: RepoIndex, *, binding: str | None = None) -> Target:
    """A named import's member within its resolved module: the top-level
    node named like it, else a re-export (`export ... from`, recorded as an
    import of the re-exporting file) for that name, following at most 5
    hops, cycle-safe. `default` follows decision 6's fallback chain instead.
    """
    if name == "default":
        return _js_default_target(module_file, binding, index)

    current = module_file
    current_name = name
    visited: set[str] = set()
    for _ in range(6):
        if current in visited:
            break
        visited.add(current)

        top_level = index.top_level_named(current, current_name)
        if top_level is not None:
            return Target(node_id=top_level.id)

        rebind = index.file_level_import(current, current_name)
        if rebind is None or not rebind.imported_name or rebind.imported_name == "*":
            break
        next_module = resolve_js_specifier(rebind.raw_name, current, index)
        if next_module.external_name is not None:
            return Target(external_name=next_module.external_name)
        if next_module.file_path is None:
            break
        current = next_module.file_path
        current_name = rebind.imported_name

    file_node = index.file_node(module_file)
    return Target(node_id=file_node.id) if file_node is not None else Target()


# --- Rust -------------------------------------------------------------

_RUST_MODULE_FILENAMES = ("mod.rs", "lib.rs", "main.rs")


def _rust_crate_root(path: str, index: RepoIndex) -> str:
    """The nearest ancestor directory containing `lib.rs` or `main.rs`
    (design.md decision 6); the repository root if none is found."""
    parts = path.split("/")[:-1]
    for i in range(len(parts), -1, -1):
        d = "/".join(parts[:i])
        lib = f"{d}/lib.rs" if d else "lib.rs"
        main = f"{d}/main.rs" if d else "main.rs"
        if index.has_file(lib) or index.has_file(main):
            return d
    return ""


def _rust_join(dir_path: str, name: str) -> str:
    """Join a module directory and a bare file/dir name, without a leading
    `/` when `dir_path` is the repository root (`""`)."""
    return f"{dir_path}/{name}" if dir_path else name


def _rust_find_module_file(dir_path: str, index: RepoIndex) -> str | None:
    for name in _RUST_MODULE_FILENAMES:
        candidate = _rust_join(dir_path, name)
        if index.has_file(candidate):
            return candidate
    return None


def _rust_module_dir(file_path: str) -> str:
    """The directory a child of this file's module would live in: its own
    directory for `mod.rs`/`lib.rs`/`main.rs`, or a directory named after
    its stem for a plain leaf file (Rust 2018's `foo.rs` + `foo/bar.rs`)."""
    dirname, _, filename = file_path.rpartition("/")
    stem = filename.removesuffix(".rs")
    if stem in ("mod", "lib", "main"):
        return dirname
    return f"{dirname}/{stem}" if dirname else stem


def _rust_parent_module_file(file_path: str, index: RepoIndex) -> str | None:
    dirname, _, filename = file_path.rpartition("/")
    stem = filename.removesuffix(".rs")
    if stem in ("mod", "lib", "main"):
        if not dirname:
            return None  # already at the crate root; no `super`
        parent_dir = dirname.rpartition("/")[0] if "/" in dirname else ""
        return _rust_find_module_file(parent_dir, index)
    return _rust_find_module_file(dirname, index)


def resolve_rust_path(raw_name: str, importer_path: str, index: RepoIndex) -> ModuleResolution:
    """Walk a `::`-joined Rust path's segments through the module tree.
    `raw_name` is the path's module part -- everything but the final
    segment, which the parser records separately as `imported_name`/`head`
    and which `resolve_rust_member` tries as a submodule before an item
    (design.md decision 6). `crate` resolves from the crate root, `self`
    from the current module, `super` from its parent; any other first
    segment resolves as a child module of the current module, or else is
    external, named by that segment."""
    if not raw_name:
        return ModuleResolution(file_path=importer_path)

    segments = raw_name.split("::")
    first, *rest = segments

    current: str | None
    if first == "crate":
        current = _rust_find_module_file(_rust_crate_root(importer_path, index), index)
    elif first == "self":
        current = importer_path
    elif first == "super":
        current = _rust_parent_module_file(importer_path, index)
    else:
        child_dir = _rust_module_dir(importer_path)
        rs, mod = _rust_join(child_dir, f"{first}.rs"), _rust_join(child_dir, f"{first}/mod.rs")
        if index.has_file(rs):
            current = rs
        elif index.has_file(mod):
            current = mod
        else:
            return ModuleResolution(external_name=first)

    if current is None:
        return ModuleResolution()

    for seg in rest:
        child_dir = _rust_module_dir(current)
        rs, mod = _rust_join(child_dir, f"{seg}.rs"), _rust_join(child_dir, f"{seg}/mod.rs")
        if index.has_file(rs):
            current = rs
        elif index.has_file(mod):
            current = mod
        else:
            return ModuleResolution()

    return ModuleResolution(file_path=current)


def resolve_rust_member(module_file: str, name: str, index: RepoIndex) -> Target:
    """A Rust path's final segment: `name` as a submodule of `module_file`
    first, then as an item it defines (design.md decision 6)."""
    child_dir = _rust_module_dir(module_file)
    rs, mod = _rust_join(child_dir, f"{name}.rs"), _rust_join(child_dir, f"{name}/mod.rs")
    if index.has_file(rs):
        return Target(node_id=index.file_node_id(rs))
    if index.has_file(mod):
        return Target(node_id=index.file_node_id(mod))
    item = index.top_level_named(module_file, name)
    if item is not None:
        return Target(node_id=item.id)
    file_node = index.file_node(module_file)
    return Target(node_id=file_node.id) if file_node is not None else Target()


# --- Class/type members (Python, JS/TS direct children; Rust by scope path)


def class_member(
    node: ResolveNode,
    name: str,
    index: RepoIndex,
    resolved_bases: dict[str, list[str]] | None = None,
) -> ResolveNode | None:
    """A class-like node's member, checking its own members first and then
    its bases' (breadth-first, cycle-safe). `resolved_bases` maps a class
    node ID to its resolved base node IDs -- supplied by group 5's tiers
    once `inherits` refs are resolved; omitted, only direct members count.
    """
    seen = {node.id}
    queue = [node]
    while queue:
        current = queue.pop(0)
        pool = (
            index.rust_type_members(current.name)
            if family_of(current.language) == "rust"
            else index.children(current.id)
        )
        match = next((m for m in pool if m.name == name), None)
        if match is not None:
            return match
        for base_id in (resolved_bases or {}).get(current.id, []):
            if base_id not in seen:
                seen.add(base_id)
                base_node = index.node(base_id)
                if base_node is not None:
                    queue.append(base_node)
    return None


# --- Tiers and outcomes --------------------------------------------------
#
# Every ref gets exactly one `Outcome` (design.md decision 7). Resolution
# runs in three passes over the whole repository, each depending on the
# last: imports first (module + member resolution alone, independent of
# everything else); then inherits (needs import outcomes for tier 3, and
# produces `resolved_bases` -- the base-class edges tier 1 walks); then
# calls (needs both).


class ResolveStatus(StrEnum):
    RESOLVED = "resolved"
    EXTERNAL = "external"
    AMBIGUOUS = "ambiguous"
    TOO_AMBIGUOUS = "too_ambiguous"
    FAILED = "failed"


class Confidence(StrEnum):
    EXACT = "exact"
    IMPORTED = "imported"
    UNIQUE_NAME = "unique_name"
    AMBIGUOUS = "ambiguous"


def external_node_id(name: str) -> str:
    return f"external::{name}"


@dataclass(frozen=True, slots=True)
class Outcome:
    """One ref's resolution outcome. `source_id` overrides the ref's own
    origin as the edge source -- used only by `for_type` (decision 4): a
    Rust trait impl's `inherits` edge originates from the implementing
    type, not the file that wrote the ref."""

    status: ResolveStatus
    target_id: str | None = None
    confidence: Confidence | None = None
    candidates: tuple[str, ...] = ()
    source_id: str | None = None


_CLASS_LIKE_KINDS = frozenset({Kind.CLASS, Kind.STRUCT, Kind.TRAIT, Kind.INTERFACE, Kind.ENUM})
_TYPE_KINDS = frozenset({Kind.STRUCT, Kind.ENUM, Kind.TYPE_ALIAS, Kind.TRAIT})
_CANDIDATE_KINDS: dict[RefKind, frozenset[Kind]] = {
    RefKind.CALLS: frozenset({Kind.FUNCTION, Kind.METHOD, Kind.CLASS, Kind.STRUCT}),
    RefKind.INHERITS: frozenset({Kind.CLASS, Kind.INTERFACE, Kind.TRAIT}),
}

# Method names of each family's builtin types, skipped by a repo-wide (not
# narrowed) tier-4 lookup for an attribute call on an unresolved receiver
# (design.md decision 5's tail stoplist) -- not exhaustive, just enough that
# `config.get(...)`/`match.get(...)` don't guess a repository class's `get`.
_TAIL_STOPLIST_BY_FAMILY: dict[str, frozenset[str]] = {
    "python": frozenset({"get", "append", "items", "join", "keys", "values", "split", "strip", "pop", "update", "add"}),
    "javascript": frozenset({"push", "map", "then", "forEach", "filter", "reduce", "join"}),
    "rust": frozenset({"unwrap", "clone", "iter", "expect", "into_iter"}),
}

_SKIP_TO_TIER4 = object()


def _chain_segments(raw_name: str, family: str | None) -> list[str]:
    """Split a call/inherits ref's `raw_name` into its dotted chain -- `.`
    for Python/JS attribute chains, `::` for a Rust path (falling back to
    `.` for a Rust method-call chain like `self.label`, which never mixes
    the two separators)."""
    sep = "::" if family == "rust" and "::" in raw_name else "."
    return [s for s in raw_name.split(sep) if s]


def _is_self_receiver(head: str, raw_name: str, family: str | None) -> bool:
    """Whether `head` is a tier-1 self-like receiver (design.md decision 5).
    Rust reuses the `self` token for both an instance receiver (`self.x`,
    dot-separated) and a module-relative path root (`self::x`, `::`-
    separated) -- only the former is tier 1; the latter is tier 3's
    current-module path. `Self` (capitalized) only ever names the enclosing
    type, so it is tier 1 either way."""
    if family == "rust":
        if head == "self":
            return "::" not in raw_name
        return head == "Self"
    return head in ("self", "cls", "this")


def _enclosing_class(origin: ResolveNode, index: RepoIndex) -> ResolveNode | None:
    node = index.node(origin.parent_id) if origin.parent_id else None
    while node is not None:
        if node.kind in _CLASS_LIKE_KINDS:
            return node
        node = index.node(node.parent_id) if node.parent_id else None
    return None


def _tier1(
    origin: ResolveNode,
    remaining: list[str],
    index: RepoIndex,
    resolved_bases: dict[str, list[str]],
    family: str | None,
) -> ResolveNode | None:
    """`self`/`cls`/`this`/`Self` -> a member of the enclosing type or its
    resolved bases (design.md decision 5). A chain longer than one member
    doesn't apply here at all -- the caller falls through to tier 4."""
    if len(remaining) != 1:
        return None
    member_name = remaining[0]
    if family == "rust":
        scope, _, _ = origin.qualified_name.rpartition(".")
        if not scope:
            return None
        type_name = scope.partition("<")[0]
        matches = [m for m in index.rust_type_members(type_name) if m.name == member_name]
        return matches[0] if matches else None
    cls = _enclosing_class(origin, index)
    return class_member(cls, member_name, index, resolved_bases) if cls is not None else None


def _tier2(
    origin: ResolveNode, head: str, remaining: list[str], index: RepoIndex, resolved_bases: dict[str, list[str]]
) -> ResolveNode | object | None:
    """A head defined in an enclosing scope (design.md decision 5): the
    origin itself, then each ancestor up to the file, skipping a class-like
    ancestor's children unless it is the origin. Returns the resolved node,
    `None` for a plain miss (falls through to tier 3), or `_SKIP_TO_TIER4`
    when the chain continues past a definition that isn't a container (a
    local shadowing a same-named fixture/helper) -- that specific case skips
    tier 3 entirely, per decision 5."""
    current: ResolveNode | None = origin
    while current is not None:
        if current.id == origin.id or current.kind not in _CLASS_LIKE_KINDS:
            match = next((c for c in index.children(current.id) if c.name == head), None)
            if match is not None:
                node = match
                for seg in remaining:
                    if node.kind not in _CLASS_LIKE_KINDS:
                        return _SKIP_TO_TIER4
                    member = class_member(node, seg, index, resolved_bases)
                    if member is None:
                        return _SKIP_TO_TIER4
                    node = member
                return node
        current = index.node(current.parent_id) if current.parent_id else None
    return None


def _module_member(family: str | None, module_file: str, name: str, index: RepoIndex) -> Target:
    if family == "python":
        return resolve_python_member(module_file, name, index)
    if family == "javascript":
        return resolve_js_member(module_file, name, index)
    if family == "rust":
        return resolve_rust_member(module_file, name, index)
    return Target()


def _module_member_strict(family: str | None, module_file: str, name: str, index: RepoIndex) -> Target | None:
    """Like `_module_member`, but `None` when `name` isn't genuinely a
    submodule, top-level node, or re-export of `module_file` -- unlike an
    import ref's own member, a wildcard-tried bare name (design.md decision
    5) must not fall back to "the module's own file node" just because the
    module exists; every module-member function uses that exact fallback
    signature when the name isn't found."""
    target = _module_member(family, module_file, name, index)
    if target.external_name is not None:
        return target
    if target.node_id is not None and target.node_id != index.file_node_id(module_file):
        return target
    return None


def _walk_from_module_target(
    outcome: Outcome, family: str | None, segments: list[str], index: RepoIndex, resolved_bases: dict[str, list[str]]
) -> Outcome:
    """Walk an already-resolved import/wildcard target through `segments`:
    a `file`-kind target is walked by the language's module-member rules
    (a further submodule or top-level node), a class-like target by its
    members. Landing on an external target at any point short-circuits the
    rest of the chain to that same external node (design.md decision 7:
    "the ref still records the external target")."""
    current = outcome
    for seg in segments:
        if current.status == ResolveStatus.EXTERNAL:
            return Outcome(ResolveStatus.EXTERNAL, target_id=current.target_id, confidence=Confidence.IMPORTED)
        if current.status != ResolveStatus.RESOLVED or current.target_id is None:
            return Outcome(ResolveStatus.FAILED)
        node = index.node(current.target_id)
        if node is None:
            return Outcome(ResolveStatus.FAILED)
        if node.kind == Kind.FILE:
            # Every module-member function falls back to the module's own file
            # node when the name isn't found there, so `target.node_id` is
            # never None here -- only an external re-export target can be.
            target = _module_member(family, node.file_path, seg, index)
            if target.external_name is not None:
                current = Outcome(ResolveStatus.EXTERNAL, target_id=external_node_id(target.external_name))
            else:
                current = Outcome(ResolveStatus.RESOLVED, target_id=target.node_id)
        elif node.kind in _CLASS_LIKE_KINDS:
            member = class_member(node, seg, index, resolved_bases)
            if member is None:
                return Outcome(ResolveStatus.FAILED)
            current = Outcome(ResolveStatus.RESOLVED, target_id=member.id)
        else:
            return Outcome(ResolveStatus.FAILED)
    if current.status in (ResolveStatus.RESOLVED, ResolveStatus.EXTERNAL) and current.target_id is not None:
        return Outcome(current.status, target_id=current.target_id, confidence=Confidence.IMPORTED)
    return Outcome(ResolveStatus.FAILED)


def _consumed_prefix_len(bound: ResolveRef, family: str | None) -> int:
    """How many of a call chain's remaining segments a plain Python import
    already accounts for. `import a.b.c` binds only `a` (`head == "a"`, the
    raw name's own first component) so a later `a.b.c.f()` must still walk
    `b` and `c`; `import a.b as x` binds the whole `a.b` under an unrelated
    name, so `x.f()` walks nothing extra. From-imports and other languages
    never have this redundancy (their `head` names exactly what the import
    bound, in full)."""
    if family != "python" or bound.imported_name:
        return 0
    components = [c for c in bound.raw_name.split(".") if c]
    if components and bound.head == components[0]:
        return len(components) - 1
    return 0


def _tier3(
    origin: ResolveNode,
    head: str,
    remaining: list[str],
    raw_name: str,
    family: str | None,
    index: RepoIndex,
    import_outcomes: dict[int, Outcome],
    resolved_bases: dict[str, list[str]],
) -> Outcome | None:
    """A head bound by a visible import, a wildcard-imported module, or (for
    Rust, since `mod`s introduce no import ref) a path from the current
    module (design.md decision 5). Returns `None` for a plain miss, which
    falls through to tier 4."""
    bound = index.visible_import(origin.id, head)
    if bound is not None:
        outcome = import_outcomes.get(bound.id)
        if outcome is None:
            return Outcome(ResolveStatus.FAILED)
        consumed = _consumed_prefix_len(bound, family)
        return _walk_from_module_target(outcome, family, remaining[consumed:], index, resolved_bases)

    if family == "rust" and "::" in raw_name:
        segments = raw_name.split("::")
        module = resolve_rust_path("::".join(segments[:-1]), origin.file_path, index)
        if module.external_name is not None:
            return Outcome(
                ResolveStatus.EXTERNAL, target_id=external_node_id(module.external_name), confidence=Confidence.IMPORTED
            )
        if module.file_path is not None:
            # resolve_rust_member never produces an external target -- only
            # resolve_rust_path (checked above) can name an external crate.
            target = resolve_rust_member(module.file_path, segments[-1], index)
            return Outcome(ResolveStatus.RESOLVED, target_id=target.node_id, confidence=Confidence.IMPORTED)

    for wref in index.visible_wildcards(origin.id):
        w_outcome = import_outcomes.get(wref.id)
        if w_outcome is None or w_outcome.status != ResolveStatus.RESOLVED or w_outcome.target_id is None:
            continue
        # A wildcard import's own outcome always targets a file node (Phase A
        # resolves `imported_name == "*"` straight to the module's file node).
        module_node = index.node(w_outcome.target_id)
        if module_node is None:
            continue
        head_target = _module_member_strict(family, module_node.file_path, head, index)
        if head_target is None:
            continue
        if head_target.external_name is not None:
            result = Outcome(
                ResolveStatus.EXTERNAL,
                target_id=external_node_id(head_target.external_name),
                confidence=Confidence.IMPORTED,
            )
        else:
            result = Outcome(ResolveStatus.RESOLVED, target_id=head_target.node_id, confidence=Confidence.IMPORTED)
        if remaining:
            result = _walk_from_module_target(result, family, remaining, index, resolved_bases)
        if result.status in (ResolveStatus.RESOLVED, ResolveStatus.EXTERNAL):
            return result
    return None


def _visible_classes(file_path: str, index: RepoIndex, import_outcomes: dict[int, Outcome]) -> list[ResolveNode]:
    """Classes a file defines or imports (design.md decision 5's tier-4
    narrowing): every class-like node in the file, plus every class-like
    node an import in the file resolves to."""
    seen: dict[str, ResolveNode] = {}
    for n in index.nodes_by_file(file_path):
        if n.kind in _CLASS_LIKE_KINDS:
            seen[n.id] = n
    for r in index.imports_in_file(file_path):
        outcome = import_outcomes.get(r.id)
        if outcome is not None and outcome.status == ResolveStatus.RESOLVED and outcome.target_id is not None:
            node = index.node(outcome.target_id)
            if node is not None and node.kind in _CLASS_LIKE_KINDS:
                seen[node.id] = node
    return list(seen.values())


def _outcome_from_matches(node_ids: set[str]) -> Outcome:
    ids = sorted(node_ids)
    if not ids:
        return Outcome(ResolveStatus.FAILED)
    if len(ids) == 1:
        return Outcome(ResolveStatus.RESOLVED, target_id=ids[0], confidence=Confidence.UNIQUE_NAME)
    if len(ids) <= 5:
        return Outcome(ResolveStatus.AMBIGUOUS, confidence=Confidence.AMBIGUOUS, candidates=tuple(ids))
    return Outcome(ResolveStatus.TOO_AMBIGUOUS, candidates=tuple(ids[:20]))


def _tier45(
    origin: ResolveNode,
    last: str,
    apply_stoplist: bool,
    candidate_kinds: frozenset[Kind],
    family: str | None,
    index: RepoIndex,
    import_outcomes: dict[int, Outcome],
    resolved_bases: dict[str, list[str]],
    *,
    narrow: bool,
) -> Outcome:
    """Unique name / ambiguous / too ambiguous / failed (design.md decision
    5): candidates narrowed to the file's visible classes first, repo-wide
    only when that set is empty."""
    if not candidate_kinds:
        return Outcome(ResolveStatus.FAILED)

    if narrow:
        matches = {
            m.id
            for cls in _visible_classes(origin.file_path, index, import_outcomes)
            for m in [class_member(cls, last, index, resolved_bases)]
            if m is not None and m.kind in candidate_kinds
        }
        if matches:
            return _outcome_from_matches(matches)

    if apply_stoplist and last in _TAIL_STOPLIST_BY_FAMILY.get(family or "", frozenset()):
        return Outcome(ResolveStatus.FAILED)

    pool = {n.id for n in index.by_name(last) if family_of(n.language) == family and n.kind in candidate_kinds}
    return _outcome_from_matches(pool)


def _resolve_chain(
    origin: ResolveNode,
    chain: list[str],
    raw_name: str,
    candidate_kinds: frozenset[Kind],
    family: str | None,
    index: RepoIndex,
    import_outcomes: dict[int, Outcome],
    resolved_bases: dict[str, list[str]],
    *,
    apply_stoplist: bool,
    allow_tier1: bool,
    narrow: bool,
) -> Outcome:
    head, *remaining = chain
    last = chain[-1]

    if allow_tier1 and _is_self_receiver(head, raw_name, family):
        member = _tier1(origin, remaining, index, resolved_bases, family)
        if member is not None:
            return Outcome(ResolveStatus.RESOLVED, target_id=member.id, confidence=Confidence.EXACT)
        return _tier45(
            origin, last, False, candidate_kinds, family, index, import_outcomes, resolved_bases, narrow=narrow
        )

    tier2 = _tier2(origin, head, remaining, index, resolved_bases)
    if isinstance(tier2, ResolveNode):
        return Outcome(ResolveStatus.RESOLVED, target_id=tier2.id, confidence=Confidence.EXACT)
    if tier2 is _SKIP_TO_TIER4:
        return _tier45(
            origin, last, apply_stoplist, candidate_kinds, family, index, import_outcomes, resolved_bases, narrow=narrow
        )

    tier3 = _tier3(origin, head, remaining, raw_name, family, index, import_outcomes, resolved_bases)
    if tier3 is not None:
        return tier3

    return _tier45(
        origin, last, apply_stoplist, candidate_kinds, family, index, import_outcomes, resolved_bases, narrow=narrow
    )


def _resolve_type_name(
    name: str, origin: ResolveNode, index: RepoIndex, import_outcomes: dict[int, Outcome]
) -> Outcome:
    """`for_type` resolution (design.md decision 4): the same scope/import/
    unique-name tiers as any head, restricted to type kinds, with no tier 1
    (there is no receiver) and no narrowing (there is no chain to narrow
    by). `name` may itself be a qualified path (Rust `crate::model::Foo`),
    so it is split into a chain exactly like a ref's `raw_name`."""
    family = family_of(origin.language)
    chain = _chain_segments(name, family)
    if not chain:
        return Outcome(ResolveStatus.FAILED)
    outcome = _resolve_chain(
        origin,
        chain,
        name,
        _TYPE_KINDS,
        family,
        index,
        import_outcomes,
        {},
        apply_stoplist=False,
        allow_tier1=False,
        narrow=False,
    )
    if outcome.status == ResolveStatus.RESOLVED and outcome.target_id is not None:
        node = index.node(outcome.target_id)
        if node is None or node.kind not in _TYPE_KINDS:
            return Outcome(ResolveStatus.FAILED)
    return outcome


def _outcome_from_module(
    module: ModuleResolution, imported_name: str | None, head: str | None, index: RepoIndex, *, family: str
) -> Outcome:
    if module.external_name is not None:
        return Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id(module.external_name), confidence=Confidence.IMPORTED
        )
    if module.file_path is None:
        return Outcome(ResolveStatus.FAILED)

    if not imported_name or imported_name == "*":
        node_id = index.file_node_id(module.file_path)
        return (
            Outcome(ResolveStatus.RESOLVED, target_id=node_id, confidence=Confidence.IMPORTED)
            if node_id
            else Outcome(ResolveStatus.FAILED)
        )

    if family == "python":
        target = resolve_python_member(module.file_path, imported_name, index)
    elif family == "javascript":
        target = resolve_js_member(module.file_path, imported_name, index, binding=head)
    else:
        target = resolve_rust_member(module.file_path, imported_name, index)

    # Every module-member function falls back to the module's own file node
    # when the name isn't found, so `target.node_id` is never None here.
    if target.external_name is not None:
        return Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id(target.external_name), confidence=Confidence.IMPORTED
        )
    return Outcome(ResolveStatus.RESOLVED, target_id=target.node_id, confidence=Confidence.IMPORTED)


def _resolve_import_ref(ref: ResolveRef, index: RepoIndex) -> Outcome:
    """Every `imports` ref resolves directly through decision 6's module and
    member resolution -- it never goes through the call/inherits tiers,
    since its own `head`/`imported_name` already say exactly what it binds
    and to what."""
    origin = index.node(ref.from_node_id)
    if origin is None:
        return Outcome(ResolveStatus.FAILED)
    family = family_of(origin.language)
    if family == "python":
        module = resolve_python_module(ref.raw_name, origin.file_path, index)
    elif family == "javascript":
        module = resolve_js_specifier(ref.raw_name, origin.file_path, index)
    elif family == "rust":
        module = resolve_rust_path(ref.raw_name, origin.file_path, index)
    else:
        return Outcome(ResolveStatus.FAILED)
    return _outcome_from_module(module, ref.imported_name, ref.head, index, family=family)


def _resolve_named_ref(
    ref: ResolveRef, index: RepoIndex, import_outcomes: dict[int, Outcome], resolved_bases: dict[str, list[str]]
) -> Outcome:
    """A `calls` or `inherits` ref: the ordinary tiers over its chain, plus
    (for a Rust trait impl) `for_type` resolution overriding the edge
    source (design.md decision 4)."""
    origin = index.node(ref.from_node_id)
    if origin is None:
        return Outcome(ResolveStatus.FAILED)
    family = family_of(origin.language)
    if family is None:
        return Outcome(ResolveStatus.FAILED)

    chain = _chain_segments(ref.raw_name, family)
    if not chain:
        return Outcome(ResolveStatus.FAILED)

    candidate_kinds = _CANDIDATE_KINDS.get(ref.ref_kind, frozenset())
    outcome = _resolve_chain(
        origin,
        chain,
        ref.raw_name,
        candidate_kinds,
        family,
        index,
        import_outcomes,
        resolved_bases,
        apply_stoplist=ref.ref_kind == RefKind.CALLS and len(chain) > 1,
        allow_tier1=True,
        narrow=True,
    )

    if ref.for_type:
        type_outcome = _resolve_type_name(ref.for_type, origin, index, import_outcomes)
        if type_outcome.status != ResolveStatus.RESOLVED or type_outcome.target_id is None:
            return Outcome(ResolveStatus.FAILED)
        if outcome.status in (ResolveStatus.RESOLVED, ResolveStatus.EXTERNAL, ResolveStatus.AMBIGUOUS):
            return replace(outcome, source_id=type_outcome.target_id)
    return outcome


def resolve_repo_refs(index: RepoIndex) -> dict[int, Outcome]:
    """Resolve every ref in the repository to an `Outcome` (design.md
    decision 1: whole-repo, in memory, every run). Imports resolve first
    and independently; inherits next, building `resolved_bases` for tier 1;
    calls last, since they are the only kind that can depend on either."""
    import_outcomes: dict[int, Outcome] = {
        r.id: _resolve_import_ref(r, index)
        for r in sorted(index.refs, key=lambda r: r.id)
        if r.ref_kind == RefKind.IMPORTS
    }
    outcomes: dict[int, Outcome] = dict(import_outcomes)

    resolved_bases: dict[str, list[str]] = {}
    for r in sorted((r for r in index.refs if r.ref_kind == RefKind.INHERITS), key=lambda r: r.id):
        outcome = _resolve_named_ref(r, index, import_outcomes, resolved_bases={})
        outcomes[r.id] = outcome
        if outcome.status == ResolveStatus.RESOLVED and outcome.target_id is not None:
            source_id = outcome.source_id or r.from_node_id
            resolved_bases.setdefault(source_id, []).append(outcome.target_id)

    for r in sorted((r for r in index.refs if r.ref_kind == RefKind.CALLS), key=lambda r: r.id):
        outcomes[r.id] = _resolve_named_ref(r, index, import_outcomes, resolved_bases)

    return outcomes
