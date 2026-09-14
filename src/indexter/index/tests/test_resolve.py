from indexter.db.connection import open_db
from indexter.index.resolve import (
    Confidence,
    ModuleResolution,
    Outcome,
    RepoIndex,
    ResolveNode,
    ResolveRef,
    ResolveStatus,
    Target,
    class_member,
    external_node_id,
    family_of,
    load_repo_index,
    resolve_js_member,
    resolve_js_specifier,
    resolve_python_member,
    resolve_python_module,
    resolve_repo_refs,
    resolve_rust_member,
    resolve_rust_path,
)
from indexter.index.tests.conftest import sync_source
from indexter.parse.models import Kind, RefKind

_counter = iter(range(1, 1_000_000))


def n(
    node_id: str,
    kind: Kind,
    name: str,
    file_path: str,
    *,
    language: str | None = "python",
    parent_id: str | None = None,
    qualified_name: str | None = None,
) -> ResolveNode:
    return ResolveNode(
        id=node_id,
        kind=kind,
        name=name,
        qualified_name=qualified_name if qualified_name is not None else name,
        file_path=file_path,
        language=language,
        parent_id=parent_id,
    )


def file_node(path: str, language: str | None = "python") -> ResolveNode:
    return n(f"{path}::#file", Kind.FILE, "", path, language=language, qualified_name=path)


def r(
    from_node_id: str,
    raw_name: str,
    *,
    head: str | None = None,
    imported_name: str | None = None,
    for_type: str | None = None,
    ref_kind: RefKind = RefKind.IMPORTS,
) -> ResolveRef:
    return ResolveRef(
        id=next(_counter),
        from_node_id=from_node_id,
        raw_name=raw_name,
        head=head,
        imported_name=imported_name,
        for_type=for_type,
        ref_kind=ref_kind,
        line=1,
        col=1,
    )


class TestFamilyOf:
    def test_maps_javascript_and_typescript_to_one_family(self):
        assert family_of("javascript") == family_of("typescript") == "javascript"

    def test_maps_rust_and_python_to_themselves(self):
        assert family_of("rust") == "rust"
        assert family_of("python") == "python"

    def test_unknown_or_missing_language_has_no_family(self):
        assert family_of("css") is None
        assert family_of(None) is None


class TestPythonModuleResolution:
    def test_source_layout_needs_no_configuration(self):
        cli = file_node("src/app/cli.py")
        config = file_node("src/app/config.py")
        load = n("src/app/config.py::load#function", Kind.FUNCTION, "load", "src/app/config.py", parent_id=config.id)
        index = RepoIndex([cli, config, load], [])

        resolution = resolve_python_module("app.config", "src/app/cli.py", index)

        assert resolution == ModuleResolution(file_path="src/app/config.py")
        assert resolve_python_member("src/app/config.py", "load", index) == Target(node_id=load.id)

    def test_relative_import(self):
        a = file_node("pkg/sub/a.py")
        util = file_node("pkg/util.py")
        helper = n("pkg/util.py::helper#function", Kind.FUNCTION, "helper", "pkg/util.py", parent_id=util.id)
        index = RepoIndex([a, util, helper], [])

        resolution = resolve_python_module("..util", "pkg/sub/a.py", index)

        assert resolution == ModuleResolution(file_path="pkg/util.py")
        assert resolve_python_member("pkg/util.py", "helper", index) == Target(node_id=helper.id)

    def test_package_reexport_is_followed(self):
        init = file_node("pkg/__init__.py")
        core = file_node("pkg/core.py")
        engine = n("pkg/core.py::Engine#class", Kind.CLASS, "Engine", "pkg/core.py", parent_id=core.id)
        reexport = r(init.id, ".core", head="Engine", imported_name="Engine")
        index = RepoIndex([init, core, engine], [reexport])

        assert resolve_python_member("pkg/__init__.py", "Engine", index) == Target(node_id=engine.id)

    def test_imported_submodule_resolves_to_its_file_node(self):
        pkg = file_node("pkg/__init__.py")
        util = file_node("pkg/util.py")
        index = RepoIndex([pkg, util], [])

        assert resolve_python_member("pkg/__init__.py", "util", index) == Target(node_id=util.id)

    def test_name_that_is_not_a_node_falls_back_to_the_module_file_node(self):
        settings = file_node("pkg/settings.py")
        index = RepoIndex([settings], [])

        assert resolve_python_member("pkg/settings.py", "defaults", index) == Target(node_id=settings.id)

    def test_absolute_import_of_a_package_matches_its_init_file(self):
        cli = file_node("cli.py")
        pkg_init = file_node("pkg/__init__.py")
        index = RepoIndex([cli, pkg_init], [])

        assert resolve_python_module("pkg", "cli.py", index) == ModuleResolution(file_path="pkg/__init__.py")

    def test_third_party_module_is_external(self):
        index = RepoIndex([file_node("app.py")], [])

        resolution = resolve_python_module("pydantic.fields", "app.py", index)

        assert resolution == ModuleResolution(external_name="pydantic")

    def test_suffix_match_prefers_longest_shared_directory_prefix(self):
        cli = file_node("src/app/cli.py")
        same_tree = file_node("src/app/config.py")
        other_tree = file_node("vendor/app/config.py")
        index = RepoIndex([cli, same_tree, other_tree], [])

        resolution = resolve_python_module("app.config", "src/app/cli.py", index)

        assert resolution == ModuleResolution(file_path="src/app/config.py")

    def test_suffix_match_tie_break_prefers_shortest_path(self):
        cli = file_node("cli.py")
        nested = file_node("a/b/config.py")
        shallow = file_node("config.py")
        index = RepoIndex([cli, nested, shallow], [])

        resolution = resolve_python_module("config", "cli.py", index)

        assert resolution == ModuleResolution(file_path="config.py")

    def test_reexport_cycle_terminates(self):
        a = file_node("a.py")
        b = file_node("b.py")
        ref_a = r(a.id, "b", head="N", imported_name="N")
        ref_b = r(b.id, "a", head="N", imported_name="N")
        index = RepoIndex([a, b], [ref_a, ref_b])

        assert resolve_python_member("a.py", "N", index) == Target(node_id=a.id)

    def test_wildcard_import_module_resolves_like_any_other(self):
        pkg = file_node("app.py")
        target = file_node("util.py")
        index = RepoIndex([pkg, target], [])
        wildcard = r(pkg.id, "util", head=None, imported_name="*")

        resolution = resolve_python_module(wildcard.raw_name, "app.py", index)

        assert resolution == ModuleResolution(file_path="util.py")


class TestJavaScriptModuleResolution:
    def test_extension_inference(self):
        app = file_node("src/app.ts", language="typescript")
        view = file_node("src/view.tsx", language="typescript")
        render = n(
            "src/view.tsx::render#function",
            Kind.FUNCTION,
            "render",
            "src/view.tsx",
            language="typescript",
            parent_id=view.id,
        )
        index = RepoIndex([app, view, render], [])

        resolution = resolve_js_specifier("./view", "src/app.ts", index)

        assert resolution == ModuleResolution(file_path="src/view.tsx")
        assert resolve_js_member("src/view.tsx", "render", index) == Target(node_id=render.id)

    def test_directory_index(self):
        app = file_node("src/app.ts", language="typescript")
        index_file = file_node("src/components/index.ts", language="typescript")
        index = RepoIndex([app, index_file], [])

        resolution = resolve_js_specifier("./components", "src/app.ts", index)

        assert resolution == ModuleResolution(file_path="src/components/index.ts")

    def test_typescript_esm_js_specifier(self):
        app = file_node("app.ts", language="typescript")
        util = file_node("util.ts", language="typescript")
        index = RepoIndex([app, util], [])

        resolution = resolve_js_specifier("./util.js", "app.ts", index)

        assert resolution == ModuleResolution(file_path="util.ts")

    def test_barrel_reexport_is_followed(self):
        lib_index = file_node("lib/index.ts", language="typescript")
        client_file = file_node("lib/client.ts", language="typescript")
        client_class = n(
            "lib/client.ts::Client#class",
            Kind.CLASS,
            "Client",
            "lib/client.ts",
            language="typescript",
            parent_id=client_file.id,
        )
        reexport = r(lib_index.id, "./client", head="Client", imported_name="Client")
        index = RepoIndex([lib_index, client_file, client_class], [reexport])

        assert resolve_js_member("lib/index.ts", "Client", index) == Target(node_id=client_class.id)

    def test_scoped_package_is_external(self):
        index = RepoIndex([file_node("app.ts", language="typescript")], [])

        resolution = resolve_js_specifier("@tanstack/react-query/devtools", "app.ts", index)

        assert resolution == ModuleResolution(external_name="@tanstack/react-query")

    def test_unscoped_package_is_external_by_first_segment(self):
        index = RepoIndex([file_node("app.ts", language="typescript")], [])

        resolution = resolve_js_specifier("lodash/debounce", "app.ts", index)

        assert resolution == ModuleResolution(external_name="lodash")

    def test_commonjs_require_target_member(self):
        app = file_node("app.js", language="javascript")
        util = file_node("util.js", language="javascript")
        fmt = n(
            "util.js::format#function", Kind.FUNCTION, "format", "util.js", language="javascript", parent_id=util.id
        )
        index = RepoIndex([app, util, fmt], [])

        resolution = resolve_js_specifier("./util", "app.js", index)

        assert resolution == ModuleResolution(file_path="util.js")
        assert resolve_js_member("util.js", "format", index) == Target(node_id=fmt.id)

    def test_default_import_resolves_to_node_named_like_binding(self):
        m = file_node("m.ts", language="typescript")
        default_export = n("m.ts::App#function", Kind.FUNCTION, "App", "m.ts", language="typescript", parent_id=m.id)
        index = RepoIndex([m, default_export], [])

        assert resolve_js_member("m.ts", "default", index, binding="App") == Target(node_id=default_export.id)

    def test_default_import_falls_back_to_only_top_level_class_or_function(self):
        m = file_node("m.ts", language="typescript")
        only_class = n("m.ts::Widget#class", Kind.CLASS, "Widget", "m.ts", language="typescript", parent_id=m.id)
        index = RepoIndex([m, only_class], [])

        assert resolve_js_member("m.ts", "default", index, binding="Anything") == Target(node_id=only_class.id)

    def test_default_import_falls_back_to_file_node(self):
        m = file_node("m.ts", language="typescript")
        index = RepoIndex([m], [])

        assert resolve_js_member("m.ts", "default", index, binding="Anything") == Target(node_id=m.id)


class TestRustModuleResolution:
    def test_crate_rooted_use(self):
        main = file_node("src/main.rs", language="rust")
        client = file_node("src/net/client.rs", language="rust")
        auth_mod = file_node("src/auth/mod.rs", language="rust")
        handler = n(
            "src/auth/mod.rs::Handler#struct",
            Kind.STRUCT,
            "Handler",
            "src/auth/mod.rs",
            language="rust",
            parent_id=auth_mod.id,
        )
        index = RepoIndex([main, client, auth_mod, handler], [])

        resolution = resolve_rust_path("crate::auth", "src/net/client.rs", index)

        assert resolution == ModuleResolution(file_path="src/auth/mod.rs")
        assert resolve_rust_member("src/auth/mod.rs", "Handler", index) == Target(node_id=handler.id)

    def test_super_path(self):
        net_mod = file_node("src/net/mod.rs", language="rust")
        client = file_node("src/net/client.rs", language="rust")
        retry = n(
            "src/net/mod.rs::retry#function",
            Kind.FUNCTION,
            "retry",
            "src/net/mod.rs",
            language="rust",
            parent_id=net_mod.id,
        )
        index = RepoIndex([net_mod, client, retry], [])

        resolution = resolve_rust_path("super", "src/net/client.rs", index)

        assert resolution == ModuleResolution(file_path="src/net/mod.rs")
        assert resolve_rust_member("src/net/mod.rs", "retry", index) == Target(node_id=retry.id)

    def test_standard_library_is_external(self):
        main = file_node("src/main.rs", language="rust")
        index = RepoIndex([main], [])

        resolution = resolve_rust_path("std::collections", "src/main.rs", index)

        assert resolution == ModuleResolution(external_name="std")

    def test_associated_function_through_a_type_in_another_file(self):
        handler_struct = n("src/model.rs::Handler#struct", Kind.STRUCT, "Handler", "src/model.rs", language="rust")
        new_fn = n(
            "src/handler_impl.rs::Handler.new#method",
            Kind.METHOD,
            "new",
            "src/handler_impl.rs",
            language="rust",
            qualified_name="Handler.new",
        )
        index = RepoIndex([handler_struct, new_fn], [])

        assert class_member(handler_struct, "new", index) == new_fn

    def test_self_inside_a_trait_impl_finds_the_inherent_impls_method(self):
        foo_struct = n("src/model.rs::Foo#struct", Kind.STRUCT, "Foo", "src/model.rs", language="rust")
        label = n(
            "src/model.rs::Foo.label#method",
            Kind.METHOD,
            "label",
            "src/model.rs",
            language="rust",
            qualified_name="Foo.label",
        )
        other_type_method = n(
            "src/other.rs::Bar.label#method",
            Kind.METHOD,
            "label",
            "src/other.rs",
            language="rust",
            qualified_name="Bar.label",
        )
        index = RepoIndex([foo_struct, label, other_type_method], [])

        assert class_member(foo_struct, "label", index) == label

    def test_trait_impl_scope_path_also_matches_bare_type_name_lookup(self):
        foo_struct = n("src/model.rs::Foo#struct", Kind.STRUCT, "Foo", "src/model.rs", language="rust")
        fmt = n(
            "src/fmt.rs::Foo<Display>.fmt#method",
            Kind.METHOD,
            "fmt",
            "src/fmt.rs",
            language="rust",
            qualified_name="Foo<Display>.fmt",
        )
        index = RepoIndex([foo_struct, fmt], [])

        assert class_member(foo_struct, "fmt", index) == fmt


class TestRustTypeMembers:
    def test_a_scope_path_with_no_dot_is_never_a_type_member(self):
        stray = n("odd.rs::stray#method", Kind.METHOD, "stray", "odd.rs", language="rust", qualified_name="stray")
        index = RepoIndex([stray], [])

        assert index.rust_type_members("stray") == []


class TestClassMemberInheritance:
    def test_own_member_wins_over_a_base_with_the_same_name(self):
        base = n("a.py::Base#class", Kind.CLASS, "Base", "a.py")
        base_save = n("a.py::Base.save#method", Kind.METHOD, "save", "a.py", parent_id=base.id)
        derived = n("a.py::Derived#class", Kind.CLASS, "Derived", "a.py")
        derived_save = n("a.py::Derived.save#method", Kind.METHOD, "save", "a.py", parent_id=derived.id)
        index = RepoIndex([base, base_save, derived, derived_save], [])

        found = class_member(derived, "save", index, resolved_bases={derived.id: [base.id]})

        assert found == derived_save

    def test_falls_through_to_resolved_base_when_not_defined_locally(self):
        base = n("a.py::Base#class", Kind.CLASS, "Base", "a.py")
        save = n("a.py::Base.save#method", Kind.METHOD, "save", "a.py", parent_id=base.id)
        derived = n("a.py::Derived#class", Kind.CLASS, "Derived", "a.py")
        index = RepoIndex([base, save, derived], [])

        found = class_member(derived, "save", index, resolved_bases={derived.id: [base.id]})

        assert found == save

    def test_without_resolved_bases_only_direct_members_count(self):
        base = n("a.py::Base#class", Kind.CLASS, "Base", "a.py")
        save = n("a.py::Base.save#method", Kind.METHOD, "save", "a.py", parent_id=base.id)
        derived = n("a.py::Derived#class", Kind.CLASS, "Derived", "a.py")
        index = RepoIndex([base, save, derived], [])

        assert class_member(derived, "save", index) is None

    def test_cycle_safe(self):
        a = n("a.py::A#class", Kind.CLASS, "A", "a.py")
        b = n("a.py::B#class", Kind.CLASS, "B", "a.py")
        index = RepoIndex([a, b], [])

        # A and B name each other as a base -- must terminate, not loop.
        assert class_member(a, "missing", index, resolved_bases={a.id: [b.id], b.id: [a.id]}) is None


class TestPythonModuleResolutionEdgeCases:
    def test_suffix_matching_ignores_files_of_other_languages(self):
        real = file_node("real/config.py", language="python")
        decoy = file_node("config.py", language="javascript")
        cli = file_node("cli.py", language="python")
        index = RepoIndex([real, decoy, cli], [])

        assert resolve_python_module("real.config", "cli.py", index) == ModuleResolution(file_path="real/config.py")

    def test_relative_import_beyond_the_repository_root_resolves_to_nothing(self):
        a = file_node("a.py")
        index = RepoIndex([a], [])

        assert resolve_python_module("....way.too.far", "a.py", index) == ModuleResolution()

    def test_empty_absolute_module_resolves_to_nothing(self):
        index = RepoIndex([file_node("a.py")], [])

        assert resolve_python_module("", "a.py", index) == ModuleResolution()

    def test_root_package_member(self):
        root_init = file_node("__init__.py")
        run = n("__init__.py::run#function", Kind.FUNCTION, "run", "__init__.py", parent_id=root_init.id)
        index = RepoIndex([root_init, run], [])

        assert resolve_python_member("__init__.py", "run", index) == Target(node_id=run.id)

    def test_imported_subpackage_resolves_to_its_init_file_node(self):
        pkg = file_node("pkg/__init__.py")
        sub_pkg = file_node("pkg/sub/__init__.py")
        index = RepoIndex([pkg, sub_pkg], [])

        assert resolve_python_member("pkg/__init__.py", "sub", index) == Target(node_id=sub_pkg.id)

    def test_reexport_chase_stops_when_it_reaches_an_external_module(self):
        a = file_node("a.py")
        reexport = r(a.id, "somepkg", head="Thing", imported_name="Thing")
        index = RepoIndex([a], [reexport])

        assert resolve_python_member("a.py", "Thing", index) == Target(external_name="somepkg")

    def test_reexport_chase_stops_when_the_rebind_module_does_not_resolve(self):
        a = file_node("a.py")
        # Points at a relative module that matches no file -- not external,
        # just absent.
        dead_end = r(a.id, "..nowhere", head="Thing", imported_name="Thing")
        index = RepoIndex([a], [dead_end])

        assert resolve_python_member("a.py", "Thing", index) == Target(node_id=a.id)

    def test_relative_import_can_resolve_to_a_package(self):
        a = file_node("pkg/a.py")
        sub_init = file_node("pkg/sub/__init__.py")
        index = RepoIndex([a, sub_init], [])

        assert resolve_python_module(".sub", "pkg/a.py", index) == ModuleResolution(file_path="pkg/sub/__init__.py")

    def test_relative_import_matching_neither_a_module_nor_a_package(self):
        a = file_node("pkg/a.py")
        index = RepoIndex([a], [])

        assert resolve_python_module(".missing", "pkg/a.py", index) == ModuleResolution()


class TestJavaScriptModuleResolutionEdgeCases:
    def test_double_dot_segments_walk_up_multiple_directories(self):
        deep = file_node("src/a/b/deep.ts", language="typescript")
        shallow = file_node("src/shallow.ts", language="typescript")
        index = RepoIndex([deep, shallow], [])

        assert resolve_js_specifier("../../shallow", "src/a/b/deep.ts", index) == ModuleResolution(
            file_path="src/shallow.ts"
        )

    def test_exact_relative_path_with_extension_already_present(self):
        app = file_node("app.ts", language="typescript")
        data = file_node("data.json", language="typescript")
        index = RepoIndex([app, data], [])

        assert resolve_js_specifier("./data.json", "app.ts", index) == ModuleResolution(file_path="data.json")

    def test_relative_specifier_matching_nothing_resolves_to_nothing(self):
        app = file_node("app.ts", language="typescript")
        index = RepoIndex([app], [])

        assert resolve_js_specifier("./missing", "app.ts", index) == ModuleResolution()

    def test_named_import_missing_everywhere_falls_back_to_file_node(self):
        m = file_node("m.ts", language="typescript")
        index = RepoIndex([m], [])

        assert resolve_js_member("m.ts", "Nope", index) == Target(node_id=m.id)

    def test_reexport_cycle_terminates(self):
        a = file_node("a.ts", language="typescript")
        b = file_node("b.ts", language="typescript")
        ref_a = r(a.id, "./b", head="N", imported_name="N")
        ref_b = r(b.id, "./a", head="N", imported_name="N")
        index = RepoIndex([a, b], [ref_a, ref_b])

        assert resolve_js_member("a.ts", "N", index) == Target(node_id=a.id)

    def test_reexport_chase_stops_at_an_external_module(self):
        a = file_node("a.ts", language="typescript")
        reexport = r(a.id, "somepkg", head="Thing", imported_name="Thing")
        index = RepoIndex([a], [reexport])

        assert resolve_js_member("a.ts", "Thing", index) == Target(external_name="somepkg")

    def test_reexport_chase_stops_when_the_rebind_module_does_not_resolve(self):
        a = file_node("a.ts", language="typescript")
        dead_end = r(a.id, "./nowhere", head="Thing", imported_name="Thing")
        index = RepoIndex([a], [dead_end])

        assert resolve_js_member("a.ts", "Thing", index) == Target(node_id=a.id)


class TestRustModuleResolutionEdgeCases:
    def test_no_crate_root_anywhere_treats_first_segment_as_relative_to_repo_root(self):
        lone = file_node("lone.rs", language="rust")
        index = RepoIndex([lone], [])

        # No lib.rs/main.rs exists anywhere, so `crate` finds no module file.
        assert resolve_rust_path("crate::auth", "lone.rs", index) == ModuleResolution()

    def test_bare_path_resolves_relative_to_the_current_file(self):
        main = file_node("main.rs", language="rust")
        index = RepoIndex([main], [])

        assert resolve_rust_path("", "main.rs", index) == ModuleResolution(file_path="main.rs")

    def test_self_with_a_further_segment(self):
        lib = file_node("src/lib.rs", language="rust")
        util_mod = file_node("src/util/mod.rs", language="rust")
        index = RepoIndex([lib, util_mod], [])

        assert resolve_rust_path("self::util", "src/lib.rs", index) == ModuleResolution(file_path="src/util/mod.rs")

    def test_super_at_the_crate_root_resolves_to_nothing(self):
        main = file_node("main.rs", language="rust")
        index = RepoIndex([main], [])

        assert resolve_rust_path("super", "main.rs", index) == ModuleResolution()

    def test_bare_first_segment_resolves_as_a_sibling_mod_rs_child_module(self):
        lib = file_node("src/lib.rs", language="rust")
        auth_mod = file_node("src/auth/mod.rs", language="rust")
        index = RepoIndex([lib, auth_mod], [])

        assert resolve_rust_path("auth", "src/lib.rs", index) == ModuleResolution(file_path="src/auth/mod.rs")

    def test_bare_first_segment_resolves_as_a_sibling_plain_file_module(self):
        lib = file_node("src/lib.rs", language="rust")
        util = file_node("src/util.rs", language="rust")
        index = RepoIndex([lib, util], [])

        assert resolve_rust_path("util", "src/lib.rs", index) == ModuleResolution(file_path="src/util.rs")

    def test_bare_first_segment_from_a_root_level_plain_file(self):
        app = file_node("app.rs", language="rust")
        helpers = file_node("app/helpers.rs", language="rust")
        index = RepoIndex([app, helpers], [])

        assert resolve_rust_path("helpers", "app.rs", index) == ModuleResolution(file_path="app/helpers.rs")

    def test_super_from_within_a_mod_rs_finds_its_own_parent_module(self):
        lib = file_node("src/lib.rs", language="rust")
        auth_mod = file_node("src/auth/mod.rs", language="rust")
        index = RepoIndex([lib, auth_mod], [])

        assert resolve_rust_path("super", "src/auth/mod.rs", index) == ModuleResolution(file_path="src/lib.rs")

    def test_multi_segment_path_lands_on_a_plain_file_module(self):
        lib = file_node("src/lib.rs", language="rust")
        auth_mod = file_node("src/auth/mod.rs", language="rust")
        login = file_node("src/auth/login.rs", language="rust")
        index = RepoIndex([lib, auth_mod, login], [])

        assert resolve_rust_path("crate::auth::login", "src/lib.rs", index) == ModuleResolution(
            file_path="src/auth/login.rs"
        )

    def test_deep_segment_that_fails_to_resolve_yields_nothing(self):
        lib = file_node("src/lib.rs", language="rust")
        auth_mod = file_node("src/auth/mod.rs", language="rust")
        index = RepoIndex([lib, auth_mod], [])

        assert resolve_rust_path("crate::auth::missing", "src/lib.rs", index) == ModuleResolution()

    def test_member_finds_a_submodule_file_before_an_item(self):
        auth_mod = file_node("src/auth/mod.rs", language="rust")
        login = file_node("src/auth/login.rs", language="rust")
        index = RepoIndex([auth_mod, login], [])

        assert resolve_rust_member("src/auth/mod.rs", "login", index) == Target(node_id=login.id)

    def test_member_finds_a_submodule_directory_before_an_item(self):
        auth_mod = file_node("src/auth/mod.rs", language="rust")
        sessions_mod = file_node("src/auth/sessions/mod.rs", language="rust")
        index = RepoIndex([auth_mod, sessions_mod], [])

        assert resolve_rust_member("src/auth/mod.rs", "sessions", index) == Target(node_id=sessions_mod.id)

    def test_member_missing_everywhere_falls_back_to_the_module_file_node(self):
        auth_mod = file_node("src/auth/mod.rs", language="rust")
        index = RepoIndex([auth_mod], [])

        assert resolve_rust_member("src/auth/mod.rs", "nope", index) == Target(node_id=auth_mod.id)


def test_load_repo_index_round_trips_through_a_real_database(repo, db_path, settings, tokenizer):
    with open_db(db_path, repo=repo, settings=settings) as conn:
        sync_source(conn, "pkg/util.py", "def helper():\n    pass\n", settings, tokenizer)
        sync_source(conn, "pkg/a.py", "from .util import helper\n\nhelper()\n", settings, tokenizer)

        index = load_repo_index(conn)

    resolution = resolve_python_module(".util", "pkg/a.py", index)
    assert resolution == ModuleResolution(file_path="pkg/util.py")

    helper_node = index.top_level_named("pkg/util.py", "helper")
    assert helper_node is not None
    assert resolve_python_member("pkg/util.py", "helper", index) == Target(node_id=helper_node.id)


# --- Group 5: tiers and outcomes -----------------------------------------


class TestImportOutcomes:
    def test_plain_import_binds_the_full_dotted_module(self):
        c = file_node("a/b/c.py")
        importer = file_node("x.py")
        imp = r(importer.id, "a.b.c", head="a")
        index = RepoIndex([c, importer], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(ResolveStatus.RESOLVED, target_id=c.id, confidence=Confidence.IMPORTED)

    def test_wildcard_import_targets_the_module_itself(self):
        util = file_node("pkg/util.py")
        importer = file_node("pkg/a.py")
        imp = r(importer.id, "pkg.util", imported_name="*")
        index = RepoIndex([util, importer], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(ResolveStatus.RESOLVED, target_id=util.id, confidence=Confidence.IMPORTED)

    def test_third_party_module_is_external(self):
        importer = file_node("pkg/a.py")
        imp = r(importer.id, "pydantic.fields", head="Field", imported_name="Field")
        index = RepoIndex([importer], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id("pydantic"), confidence=Confidence.IMPORTED
        )

    def test_rust_standard_library_import_is_external(self):
        importer = file_node("src/main.rs", language="rust")
        imp = r(importer.id, "std::collections", head="HashMap", imported_name="HashMap")
        index = RepoIndex([importer], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id("std"), confidence=Confidence.IMPORTED
        )

    def test_rust_single_segment_use_is_external(self):
        importer = file_node("src/main.rs", language="rust")
        imp = r(importer.id, "serde", head="serde")
        index = RepoIndex([importer], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id("serde"), confidence=Confidence.IMPORTED
        )

    def test_relative_import_matching_nothing_fails(self):
        importer = file_node("a.py")
        imp = r(importer.id, "....nope", head="nope", imported_name="nope")
        index = RepoIndex([importer], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(ResolveStatus.FAILED)

    def test_import_ref_with_no_family_fails(self):
        importer = file_node("styles.css", language="css")
        imp = r(importer.id, "./other.css", head=None)
        index = RepoIndex([importer], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(ResolveStatus.FAILED)

    def test_import_ref_with_a_dangling_origin_fails(self):
        imp = r("nonexistent-node-id", "pkg.auth", head="auth")
        index = RepoIndex([], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(ResolveStatus.FAILED)

    def test_javascript_named_import(self):
        util = file_node("util.js", language="javascript")
        fmt = n(
            "util.js::format#function",
            Kind.FUNCTION,
            "format",
            "util.js",
            language="javascript",
            parent_id=util.id,
        )
        main = file_node("main.js", language="javascript")
        imp = r(main.id, "./util", head="format", imported_name="format")
        index = RepoIndex([util, fmt, main], [imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(ResolveStatus.RESOLVED, target_id=fmt.id, confidence=Confidence.IMPORTED)

    def test_named_import_that_is_itself_a_re_export_of_an_external_package(self):
        barrel = file_node("pkg/barrel.py")
        rebind = r(barrel.id, "external_pkg", head="Thing", imported_name="Thing")
        importer = file_node("pkg/a.py")
        imp = r(importer.id, "pkg.barrel", head="Thing", imported_name="Thing")
        index = RepoIndex([barrel, importer], [rebind, imp])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id("external_pkg"), confidence=Confidence.IMPORTED
        )


class TestTier1SelfReceiver:
    def test_method_call_on_self_resolves_exact(self):
        f = file_node("pkg/a.py")
        foo = n("pkg/a.py::Foo#class", Kind.CLASS, "Foo", "pkg/a.py", parent_id=f.id)
        validate = n("pkg/a.py::Foo.validate#method", Kind.METHOD, "validate", "pkg/a.py", parent_id=foo.id)
        caller = n("pkg/a.py::Foo.run#method", Kind.METHOD, "run", "pkg/a.py", parent_id=foo.id)
        call = r(caller.id, "self.validate", head="self", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, foo, validate, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=validate.id, confidence=Confidence.EXACT)

    def test_inherited_member_through_self_is_still_exact(self):
        f = file_node("pkg/a.py")
        base = n("pkg/a.py::Base#class", Kind.CLASS, "Base", "pkg/a.py", parent_id=f.id)
        save = n("pkg/a.py::Base.save#method", Kind.METHOD, "save", "pkg/a.py", parent_id=base.id)
        derived = n("pkg/a.py::Derived#class", Kind.CLASS, "Derived", "pkg/a.py", parent_id=f.id)
        run = n("pkg/a.py::Derived.run#method", Kind.METHOD, "run", "pkg/a.py", parent_id=derived.id)
        inherits = r(derived.id, "Base", head="Base", ref_kind=RefKind.INHERITS)
        call = r(run.id, "self.save", head="self", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, base, save, derived, run], [inherits, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=save.id, confidence=Confidence.EXACT)

    def test_chain_longer_than_one_member_falls_through_to_tier4(self):
        f = file_node("pkg/a.py")
        foo = n("pkg/a.py::Foo#class", Kind.CLASS, "Foo", "pkg/a.py", parent_id=f.id)
        run = n("pkg/a.py::Foo.run#method", Kind.METHOD, "run", "pkg/a.py", parent_id=foo.id)
        other_file = file_node("pkg/store.py")
        store_cls = n("pkg/store.py::Store#class", Kind.CLASS, "Store", "pkg/store.py", parent_id=other_file.id)
        add = n("pkg/store.py::Store.add#method", Kind.METHOD, "add", "pkg/store.py", parent_id=store_cls.id)
        call = r(run.id, "self.store.add", head="self", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, foo, run, other_file, store_cls, add], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=add.id, confidence=Confidence.UNIQUE_NAME)

    def test_self_miss_falls_through_to_tier4_and_can_still_fail(self):
        f = file_node("pkg/a.py")
        foo = n("pkg/a.py::Foo#class", Kind.CLASS, "Foo", "pkg/a.py", parent_id=f.id)
        run = n("pkg/a.py::Foo.run#method", Kind.METHOD, "run", "pkg/a.py", parent_id=foo.id)
        call = r(run.id, "self.nonexistent", head="self", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, foo, run], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_self_with_no_enclosing_class_falls_through(self):
        f = file_node("pkg/a.py")
        run = n("pkg/a.py::run#function", Kind.FUNCTION, "run", "pkg/a.py", parent_id=f.id)
        call = r(run.id, "self.validate", head="self", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, run], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_rust_self_inside_a_trait_impl_resolves_to_the_inherent_method(self):
        display_file = file_node("src/fmt.rs", language="rust")
        fmt = n(
            "src/fmt.rs::Foo<Display>.fmt#method",
            Kind.METHOD,
            "fmt",
            "src/fmt.rs",
            language="rust",
            parent_id=display_file.id,
            qualified_name="Foo<Display>.fmt",
        )
        inherent_file = file_node("src/foo.rs", language="rust")
        label = n(
            "src/foo.rs::Foo.label#method",
            Kind.METHOD,
            "label",
            "src/foo.rs",
            language="rust",
            parent_id=inherent_file.id,
            qualified_name="Foo.label",
        )
        call = r(fmt.id, "self.label", head="self", ref_kind=RefKind.CALLS)
        index = RepoIndex([display_file, fmt, inherent_file, label], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=label.id, confidence=Confidence.EXACT)

    def test_rust_self_colon_colon_path_is_not_tier1(self):
        client = file_node("src/net/client.rs", language="rust")
        helper = n(
            "src/net/client.rs::helper#function",
            Kind.FUNCTION,
            "helper",
            "src/net/client.rs",
            language="rust",
            parent_id=client.id,
        )
        run = n(
            "src/net/client.rs::run#function",
            Kind.FUNCTION,
            "run",
            "src/net/client.rs",
            language="rust",
            parent_id=client.id,
        )
        call = r(run.id, "self::helper", head="self", ref_kind=RefKind.CALLS)
        index = RepoIndex([client, helper, run], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=helper.id, confidence=Confidence.IMPORTED)

    def test_rust_bare_self_field_is_not_a_type_receiver(self):
        f = file_node("src/lib.rs", language="rust")
        run = n("src/lib.rs::run#function", Kind.FUNCTION, "run", "src/lib.rs", language="rust", parent_id=f.id)
        call = r(run.id, "value", head="value", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, run], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_rust_self_in_a_free_function_has_no_enclosing_type(self):
        f = file_node("src/lib.rs", language="rust")
        run = n("src/lib.rs::run#function", Kind.FUNCTION, "run", "src/lib.rs", language="rust", parent_id=f.id)
        call = r(run.id, "self.label", head="self", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, run], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)


class TestTier2EnclosingScope:
    def test_call_to_a_function_in_the_same_file(self):
        f = file_node("pkg/a.py")
        helper = n("pkg/a.py::helper#function", Kind.FUNCTION, "helper", "pkg/a.py", parent_id=f.id)
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        call = r(caller.id, "helper", head="helper", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, helper, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=helper.id, confidence=Confidence.EXACT)

    def test_innermost_enclosing_definition_wins(self):
        f = file_node("pkg/a.py")
        outer = n("pkg/a.py::outer#function", Kind.FUNCTION, "outer", "pkg/a.py", parent_id=f.id)
        inner_run = n("pkg/a.py::outer.run#function", Kind.FUNCTION, "run", "pkg/a.py", parent_id=outer.id)
        file_run = n("pkg/a.py::run#function", Kind.FUNCTION, "run", "pkg/a.py", parent_id=f.id)
        call = r(outer.id, "run", head="run", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, outer, inner_run, file_run], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=inner_run.id, confidence=Confidence.EXACT)

    def test_method_bodies_do_not_see_sibling_methods_by_bare_name_at_exact_confidence(self):
        f = file_node("pkg/a.py")
        foo = n("pkg/a.py::Foo#class", Kind.CLASS, "Foo", "pkg/a.py", parent_id=f.id)
        validate = n("pkg/a.py::Foo.validate#method", Kind.METHOD, "validate", "pkg/a.py", parent_id=foo.id)
        other = n("pkg/a.py::Foo.other#method", Kind.METHOD, "other", "pkg/a.py", parent_id=foo.id)
        call = r(other.id, "validate", head="validate", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, foo, validate, other], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id].confidence != Confidence.EXACT

    def test_chain_continuing_past_a_non_container_falls_through_to_tier4(self):
        f = file_node("pkg/a.py")
        fixture_fn = n("pkg/a.py::fixture_fn#function", Kind.FUNCTION, "fixture_fn", "pkg/a.py", parent_id=f.id)
        caller = n("pkg/a.py::test_it#function", Kind.FUNCTION, "test_it", "pkg/a.py", parent_id=f.id)
        other_file = file_node("pkg/parser.py")
        parser_cls = n("pkg/parser.py::Parser#class", Kind.CLASS, "Parser", "pkg/parser.py", parent_id=other_file.id)
        parse = n("pkg/parser.py::Parser.parse#method", Kind.METHOD, "parse", "pkg/parser.py", parent_id=parser_cls.id)
        call = r(caller.id, "fixture_fn.parse", head="fixture_fn", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, fixture_fn, caller, other_file, parser_cls, parse], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.RESOLVED, target_id=parse.id, confidence=Confidence.UNIQUE_NAME
        )

    def test_chain_continuing_past_a_local_class_missing_the_member_falls_through(self):
        f = file_node("pkg/a.py")
        foo = n("pkg/a.py::Foo#class", Kind.CLASS, "Foo", "pkg/a.py", parent_id=f.id)
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        call = r(caller.id, "Foo.nonexistent", head="Foo", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, foo, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_chain_through_a_locally_defined_class_member(self):
        f = file_node("pkg/a.py")
        foo = n("pkg/a.py::Foo#class", Kind.CLASS, "Foo", "pkg/a.py", parent_id=f.id)
        bar = n("pkg/a.py::Foo.bar#method", Kind.METHOD, "bar", "pkg/a.py", parent_id=foo.id)
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        call = r(caller.id, "Foo.bar", head="Foo", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, foo, bar, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=bar.id, confidence=Confidence.EXACT)


class TestTier3ImportBound:
    def test_call_through_an_import(self):
        auth = file_node("pkg/auth.py")
        login = n("pkg/auth.py::login#function", Kind.FUNCTION, "login", "pkg/auth.py", parent_id=auth.id)
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::handler#function", Kind.FUNCTION, "handler", "pkg/a.py", parent_id=a.id)
        imp = r(a.id, "pkg.auth", head="login", imported_name="login")
        call = r(caller.id, "login", head="login", ref_kind=RefKind.CALLS)
        index = RepoIndex([auth, login, a, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=login.id, confidence=Confidence.IMPORTED)

    def test_chain_continuing_past_an_imported_function_fails(self):
        auth = file_node("pkg/auth.py")
        login = n("pkg/auth.py::login#function", Kind.FUNCTION, "login", "pkg/auth.py", parent_id=auth.id)
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::handler#function", Kind.FUNCTION, "handler", "pkg/a.py", parent_id=a.id)
        imp = r(a.id, "pkg.auth", head="login", imported_name="login")
        call = r(caller.id, "login.sub", head="login", ref_kind=RefKind.CALLS)
        index = RepoIndex([auth, login, a, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_import_bound_module_member_reexports_to_an_external_package(self):
        util = file_node("pkg/util.py")
        rebind = r(util.id, "somewhere", head="thing", imported_name="thing_impl")
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=a.id)
        imp = r(a.id, "pkg.util", head="u")
        call = r(caller.id, "u.thing", head="u", ref_kind=RefKind.CALLS)
        index = RepoIndex([util, a, caller], [rebind, imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id("somewhere"), confidence=Confidence.IMPORTED
        )

    def test_aliased_import(self):
        auth = file_node("pkg/auth.py")
        login = n("pkg/auth.py::login#function", Kind.FUNCTION, "login", "pkg/auth.py", parent_id=auth.id)
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::handler#function", Kind.FUNCTION, "handler", "pkg/a.py", parent_id=a.id)
        imp = r(a.id, "pkg.auth", head="sign_in", imported_name="login")
        call = r(caller.id, "sign_in", head="sign_in", ref_kind=RefKind.CALLS)
        index = RepoIndex([auth, login, a, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=login.id, confidence=Confidence.IMPORTED)

    def test_module_qualified_call(self):
        auth = file_node("pkg/auth.py")
        login = n("pkg/auth.py::login#function", Kind.FUNCTION, "login", "pkg/auth.py", parent_id=auth.id)
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::handler#function", Kind.FUNCTION, "handler", "pkg/a.py", parent_id=a.id)
        imp = r(a.id, "pkg.auth", head="pkg")
        call = r(caller.id, "pkg.auth.login", head="pkg", ref_kind=RefKind.CALLS)
        index = RepoIndex([auth, login, a, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=login.id, confidence=Confidence.IMPORTED)

    def test_function_local_import_shadows_a_file_level_one(self):
        mod_a = file_node("pkg/a.py")
        load_a = n("pkg/a.py::load#function", Kind.FUNCTION, "load", "pkg/a.py", parent_id=mod_a.id)
        mod_b = file_node("pkg/b.py")
        load_b = n("pkg/b.py::load#function", Kind.FUNCTION, "load", "pkg/b.py", parent_id=mod_b.id)
        user_file = file_node("pkg/user.py")
        file_import = r(user_file.id, "pkg.a", head="load", imported_name="load")
        foo = n("pkg/user.py::foo#function", Kind.FUNCTION, "foo", "pkg/user.py", parent_id=user_file.id)
        local_import = r(foo.id, "pkg.b", head="load", imported_name="load")
        call = r(foo.id, "load", head="load", ref_kind=RefKind.CALLS)
        index = RepoIndex(
            [mod_a, load_a, mod_b, load_b, user_file, foo],
            [file_import, local_import, call],
        )

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=load_b.id, confidence=Confidence.IMPORTED)

    def test_javascript_commonjs_require(self):
        util = file_node("util.js", language="javascript")
        fmt = n(
            "util.js::format#function",
            Kind.FUNCTION,
            "format",
            "util.js",
            language="javascript",
            parent_id=util.id,
        )
        main = file_node("main.js", language="javascript")
        caller = n("main.js::run#function", Kind.FUNCTION, "run", "main.js", language="javascript", parent_id=main.id)
        imp = r(main.id, "./util", head="util")
        call = r(caller.id, "util.format", head="util", ref_kind=RefKind.CALLS)
        index = RepoIndex([util, fmt, main, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=fmt.id, confidence=Confidence.IMPORTED)

    def test_import_target_that_is_ambiguous_fails_a_dependent_call(self):
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=a.id)
        imp = r(a.id, "pkg.nonexistent", head="thing", imported_name="thing")
        call = r(caller.id, "thing", head="thing", ref_kind=RefKind.CALLS)
        index = RepoIndex([a, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[imp.id] == Outcome(
            ResolveStatus.EXTERNAL,
            target_id=external_node_id("pkg"),
            confidence=Confidence.IMPORTED,
        )
        assert outcomes[call.id] == Outcome(
            ResolveStatus.EXTERNAL,
            target_id=external_node_id("pkg"),
            confidence=Confidence.IMPORTED,
        )


class TestTier3RustModulePaths:
    def test_super_path(self):
        net_mod = file_node("src/net/mod.rs", language="rust")
        retry = n(
            "src/net/mod.rs::retry#function",
            Kind.FUNCTION,
            "retry",
            "src/net/mod.rs",
            language="rust",
            parent_id=net_mod.id,
        )
        client = file_node("src/net/client.rs", language="rust")
        run = n(
            "src/net/client.rs::run#function",
            Kind.FUNCTION,
            "run",
            "src/net/client.rs",
            language="rust",
            parent_id=client.id,
        )
        call = r(run.id, "super::retry", head="super", ref_kind=RefKind.CALLS)
        index = RepoIndex([net_mod, retry, client, run], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=retry.id, confidence=Confidence.IMPORTED)

    def test_bare_path_after_a_child_module_declaration(self):
        auth = file_node("src/auth.rs", language="rust")
        login = n(
            "src/auth.rs::login#function",
            Kind.FUNCTION,
            "login",
            "src/auth.rs",
            language="rust",
            parent_id=auth.id,
        )
        main_file = file_node("src/main.rs", language="rust")
        caller = n(
            "src/main.rs::run#function",
            Kind.FUNCTION,
            "run",
            "src/main.rs",
            language="rust",
            parent_id=main_file.id,
        )
        call = r(caller.id, "auth::login", head="auth", ref_kind=RefKind.CALLS)
        index = RepoIndex([auth, login, main_file, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=login.id, confidence=Confidence.IMPORTED)

    def test_associated_function_through_an_imported_type_in_another_file(self):
        model = file_node("src/model.rs", language="rust")
        handler_struct = n(
            "src/model.rs::Handler#struct", Kind.STRUCT, "Handler", "src/model.rs", language="rust", parent_id=model.id
        )
        impl_file = file_node("src/handler_impl.rs", language="rust")
        new_fn = n(
            "src/handler_impl.rs::Handler.new#method",
            Kind.METHOD,
            "new",
            "src/handler_impl.rs",
            language="rust",
            parent_id=impl_file.id,
            qualified_name="Handler.new",
        )
        caller_file = file_node("src/main.rs", language="rust")
        caller = n(
            "src/main.rs::run#function", Kind.FUNCTION, "run", "src/main.rs", language="rust", parent_id=caller_file.id
        )
        imp = r(caller_file.id, "crate::model", head="Handler", imported_name="Handler")
        call = r(caller.id, "Handler::new", head="Handler", ref_kind=RefKind.CALLS)
        index = RepoIndex([model, handler_struct, impl_file, new_fn, caller_file, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=new_fn.id, confidence=Confidence.IMPORTED)

    def test_deep_path_miss_falls_through_to_tier4(self):
        main_file = file_node("src/main.rs", language="rust")
        caller = n(
            "src/main.rs::run#function", Kind.FUNCTION, "run", "src/main.rs", language="rust", parent_id=main_file.id
        )
        # "auth" is a real child module (so the first segment resolves), but it has
        # no "sub" submodule -- the miss must happen deep in the path, not at "auth".
        auth_file = file_node("src/auth.rs", language="rust")
        other_file = file_node("src/other.rs", language="rust")
        thing = n(
            "src/other.rs::thing#function",
            Kind.FUNCTION,
            "thing",
            "src/other.rs",
            language="rust",
            parent_id=other_file.id,
        )
        call = r(caller.id, "auth::sub::thing", head="auth", ref_kind=RefKind.CALLS)
        index = RepoIndex([main_file, caller, auth_file, other_file, thing], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.RESOLVED, target_id=thing.id, confidence=Confidence.UNIQUE_NAME
        )

    def test_unbound_head_that_is_not_a_real_module_is_external(self):
        main_file = file_node("src/main.rs", language="rust")
        caller = n(
            "src/main.rs::run#function", Kind.FUNCTION, "run", "src/main.rs", language="rust", parent_id=main_file.id
        )
        call = r(caller.id, "randomcrate::thing", head="randomcrate", ref_kind=RefKind.CALLS)
        index = RepoIndex([main_file, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id("randomcrate"), confidence=Confidence.IMPORTED
        )

    def test_import_bound_to_a_rust_module_continues_through_module_member(self):
        model = file_node("src/model.rs", language="rust")
        foo_struct = n(
            "src/model.rs::Foo#struct",
            Kind.STRUCT,
            "Foo",
            "src/model.rs",
            language="rust",
            parent_id=model.id,
        )
        impl_file = file_node("src/impl.rs", language="rust")
        new_fn = n(
            "src/impl.rs::Foo.new#method",
            Kind.METHOD,
            "new",
            "src/impl.rs",
            language="rust",
            parent_id=impl_file.id,
            qualified_name="Foo.new",
        )
        main_file = file_node("src/main.rs", language="rust")
        caller = n(
            "src/main.rs::run#function", Kind.FUNCTION, "run", "src/main.rs", language="rust", parent_id=main_file.id
        )
        imp = r(main_file.id, "crate::model", head="model")
        call = r(caller.id, "model::Foo::new", head="model", ref_kind=RefKind.CALLS)
        index = RepoIndex([model, foo_struct, impl_file, new_fn, main_file, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=new_fn.id, confidence=Confidence.IMPORTED)


class TestTier3Wildcards:
    def test_unbound_head_resolves_through_a_wildcard_import(self):
        util = file_node("pkg/util.py")
        helper = n("pkg/util.py::helper#function", Kind.FUNCTION, "helper", "pkg/util.py", parent_id=util.id)
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=a.id)
        wildcard = r(a.id, "pkg.util", imported_name="*")
        call = r(caller.id, "helper", head="helper", ref_kind=RefKind.CALLS)
        index = RepoIndex([util, helper, a, caller], [wildcard, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=helper.id, confidence=Confidence.IMPORTED)

    def test_wildcard_that_does_not_have_the_name_falls_through(self):
        util = file_node("pkg/util.py")
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=a.id)
        wildcard = r(a.id, "pkg.util", imported_name="*")
        other_file = file_node("pkg/thing.py")
        thing_helper = n(
            "pkg/thing.py::helper2#function", Kind.FUNCTION, "helper2", "pkg/thing.py", parent_id=other_file.id
        )
        call = r(caller.id, "helper2", head="helper2", ref_kind=RefKind.CALLS)
        index = RepoIndex([util, a, caller, other_file, thing_helper], [wildcard, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.RESOLVED, target_id=thing_helper.id, confidence=Confidence.UNIQUE_NAME
        )

    def test_external_wildcard_is_skipped(self):
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=a.id)
        wildcard = r(a.id, "somepkg", imported_name="*")
        call = r(caller.id, "whatever", head="whatever", ref_kind=RefKind.CALLS)
        index = RepoIndex([a, caller], [wildcard, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_wildcard_with_a_dangling_target_is_skipped(self):
        from indexter.index.resolve import _tier3

        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        wildcard = r(f.id, "pkg.ghost", imported_name="*")
        index = RepoIndex([f, caller], [wildcard])
        fake_outcomes = {wildcard.id: Outcome(ResolveStatus.RESOLVED, target_id="ghost-id")}

        result = _tier3(caller, "whatever", [], "whatever", "python", index, fake_outcomes, {})

        assert result is None

    def test_wildcard_member_that_is_itself_an_external_reexport(self):
        util = file_node("pkg/util.py")
        rebind = r(util.id, "somewhere_else", head="head", imported_name="head_name")
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=a.id)
        wildcard = r(a.id, "pkg.util", imported_name="*")
        call = r(caller.id, "head", head="head", ref_kind=RefKind.CALLS)
        index = RepoIndex([util, a, caller], [rebind, wildcard, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id("somewhere_else"), confidence=Confidence.IMPORTED
        )

    def test_wildcard_import_member_with_chain_continuation(self):
        util = file_node("pkg/util.py")
        foo_cls = n("pkg/util.py::Foo#class", Kind.CLASS, "Foo", "pkg/util.py", parent_id=util.id)
        bar = n("pkg/util.py::Foo.bar#method", Kind.METHOD, "bar", "pkg/util.py", parent_id=foo_cls.id)
        a = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=a.id)
        wildcard = r(a.id, "pkg.util", imported_name="*")
        call = r(caller.id, "Foo.bar", head="Foo", ref_kind=RefKind.CALLS)
        index = RepoIndex([util, foo_cls, bar, a, caller], [wildcard, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=bar.id, confidence=Confidence.IMPORTED)


class TestTier4And5NameLookup:
    def test_name_unique_in_the_repository(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        other = file_node("pkg/thing.py")
        cls = n("pkg/thing.py::Thing#class", Kind.CLASS, "Thing", "pkg/thing.py", parent_id=other.id)
        frobnicate = n(
            "pkg/thing.py::Thing.frobnicate#method",
            Kind.METHOD,
            "frobnicate",
            "pkg/thing.py",
            parent_id=cls.id,
        )
        call = r(caller.id, "something.frobnicate", head="something", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, caller, other, cls, frobnicate], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.RESOLVED, target_id=frobnicate.id, confidence=Confidence.UNIQUE_NAME
        )

    def test_two_to_five_candidates_is_ambiguous(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        nodes: list[ResolveNode] = [f, caller]
        expected_ids = []
        for i in range(3):
            fp = f"pkg/p{i}.py"
            file_n = file_node(fp)
            cls = n(f"{fp}::P#class", Kind.CLASS, "P", fp, parent_id=file_n.id)
            m = n(f"{fp}::P.parse#method", Kind.METHOD, "parse", fp, parent_id=cls.id)
            nodes += [file_n, cls, m]
            expected_ids.append(m.id)
        call = r(caller.id, "something.parse", head="something", ref_kind=RefKind.CALLS)
        index = RepoIndex(nodes, [call])

        outcomes = resolve_repo_refs(index)

        outcome = outcomes[call.id]
        assert outcome.status == ResolveStatus.AMBIGUOUS
        assert outcome.confidence == Confidence.AMBIGUOUS
        assert sorted(outcome.candidates) == sorted(expected_ids)

    def test_more_than_five_candidates_is_too_ambiguous(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        nodes: list[ResolveNode] = [f, caller]
        expected_ids = []
        for i in range(8):
            fp = f"pkg/p{i}.py"
            file_n = file_node(fp)
            cls = n(f"{fp}::P#class", Kind.CLASS, "P", fp, parent_id=file_n.id)
            m = n(f"{fp}::P.parse#method", Kind.METHOD, "parse", fp, parent_id=cls.id)
            nodes += [file_n, cls, m]
            expected_ids.append(m.id)
        call = r(caller.id, "something.parse", head="something", ref_kind=RefKind.CALLS)
        index = RepoIndex(nodes, [call])

        outcomes = resolve_repo_refs(index)

        outcome = outcomes[call.id]
        assert outcome.status == ResolveStatus.TOO_AMBIGUOUS
        assert outcome.confidence is None
        assert outcome.target_id is None
        assert sorted(outcome.candidates) == sorted(expected_ids)

    def test_candidate_list_is_capped_at_twenty(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        nodes: list[ResolveNode] = [f, caller]
        expected_ids = []
        for i in range(25):
            fp = f"pkg/p{i}.py"
            file_n = file_node(fp)
            cls = n(f"{fp}::P#class", Kind.CLASS, "P", fp, parent_id=file_n.id)
            m = n(f"{fp}::P.parse#method", Kind.METHOD, "parse", fp, parent_id=cls.id)
            nodes += [file_n, cls, m]
            expected_ids.append(m.id)
        call = r(caller.id, "something.parse", head="something", ref_kind=RefKind.CALLS)
        index = RepoIndex(nodes, [call])

        outcomes = resolve_repo_refs(index)

        outcome = outcomes[call.id]
        assert outcome.status == ResolveStatus.TOO_AMBIGUOUS
        assert len(outcome.candidates) == 20
        assert list(outcome.candidates) == sorted(expected_ids)[:20]

    def test_no_match_is_failed(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        call = r(caller.id, "nonexistent", head="nonexistent", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_resolution_never_crosses_language_families(self):
        py_file = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=py_file.id)
        js_file = file_node("web/a.js", language="javascript")
        js_fn = n(
            "web/a.js::shared#function",
            Kind.FUNCTION,
            "shared",
            "web/a.js",
            language="javascript",
            parent_id=js_file.id,
        )
        call = r(caller.id, "shared", head="shared", ref_kind=RefKind.CALLS)
        index = RepoIndex([py_file, caller, js_file, js_fn], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_tier45_with_no_candidate_kinds_fails_immediately(self):
        # imports never reach tier 4/5 in resolve_repo_refs (they resolve directly
        # through decision 6), so the guard against an unmapped ref kind's empty
        # candidate set is exercised directly here.
        from indexter.index.resolve import _tier45

        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        index = RepoIndex([f, caller], [])

        outcome = _tier45(caller, "helper", False, frozenset(), "python", index, {}, {}, narrow=True)

        assert outcome == Outcome(ResolveStatus.FAILED)


class TestNarrowing:
    def test_imported_class_disambiguates_a_method_name(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        rp_file = file_node("pkg/rustp.py")
        rust_parser = n(
            "pkg/rustp.py::RustParser#class",
            Kind.CLASS,
            "RustParser",
            "pkg/rustp.py",
            parent_id=rp_file.id,
        )
        rust_parse = n(
            "pkg/rustp.py::RustParser.parse#method", Kind.METHOD, "parse", "pkg/rustp.py", parent_id=rust_parser.id
        )
        imp = r(f.id, "pkg.rustp", head="RustParser", imported_name="RustParser")
        nodes: list[ResolveNode] = [f, caller, rp_file, rust_parser, rust_parse]
        for i in range(4):
            fp = f"pkg/other{i}.py"
            file_n = file_node(fp)
            cls = n(f"{fp}::Other#class", Kind.CLASS, "Other", fp, parent_id=file_n.id)
            m = n(f"{fp}::Other.parse#method", Kind.METHOD, "parse", fp, parent_id=cls.id)
            nodes += [file_n, cls, m]
        call = r(caller.id, "parser.parse", head="parser", ref_kind=RefKind.CALLS)
        index = RepoIndex(nodes, [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.RESOLVED, target_id=rust_parse.id, confidence=Confidence.UNIQUE_NAME
        )

    def test_inherited_method_found_through_an_imported_class(self):
        base_file = file_node("pkg/base.py")
        base_cls = n("pkg/base.py::BaseParser#class", Kind.CLASS, "BaseParser", "pkg/base.py", parent_id=base_file.id)
        base_parse = n(
            "pkg/base.py::BaseParser.parse#method", Kind.METHOD, "parse", "pkg/base.py", parent_id=base_cls.id
        )
        rp_file = file_node("pkg/rustp.py")
        rust_parser = n(
            "pkg/rustp.py::RustParser#class",
            Kind.CLASS,
            "RustParser",
            "pkg/rustp.py",
            parent_id=rp_file.id,
        )
        inherits = r(rust_parser.id, "BaseParser", head="BaseParser", ref_kind=RefKind.INHERITS)
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        imp = r(f.id, "pkg.rustp", head="RustParser", imported_name="RustParser")
        call = r(caller.id, "parser.parse", head="parser", ref_kind=RefKind.CALLS)
        index = RepoIndex(
            [base_file, base_cls, base_parse, rp_file, rust_parser, f, caller],
            [inherits, imp, call],
        )

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id].status == ResolveStatus.RESOLVED
        assert outcomes[call.id].target_id == base_parse.id

    def test_empty_narrowed_set_falls_back_to_the_repository(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        other = file_node("pkg/thing.py")
        cls = n("pkg/thing.py::Thing#class", Kind.CLASS, "Thing", "pkg/thing.py", parent_id=other.id)
        normalize = n(
            "pkg/thing.py::Thing.normalize#method",
            Kind.METHOD,
            "normalize",
            "pkg/thing.py",
            parent_id=cls.id,
        )
        # An unrelated import of a plain function shouldn't count as a "visible class".
        helper_file = file_node("pkg/helper.py")
        helper = n("pkg/helper.py::helper#function", Kind.FUNCTION, "helper", "pkg/helper.py", parent_id=helper_file.id)
        imp = r(f.id, "pkg.helper", head="helper", imported_name="helper")
        call = r(caller.id, "value.normalize", head="value", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, caller, other, cls, normalize, helper_file, helper], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(
            ResolveStatus.RESOLVED, target_id=normalize.id, confidence=Confidence.UNIQUE_NAME
        )


class TestTailStoplist:
    def test_dictionary_access_on_an_unbound_receiver_is_not_guessed(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        other = file_node("pkg/thing.py")
        cls = n("pkg/thing.py::Thing#class", Kind.CLASS, "Thing", "pkg/thing.py", parent_id=other.id)
        get = n("pkg/thing.py::Thing.get#method", Kind.METHOD, "get", "pkg/thing.py", parent_id=cls.id)
        call = r(caller.id, "config.get", head="config", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, caller, other, cls, get], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_narrowed_lookup_still_finds_a_stoplisted_name(self):
        store_file = file_node("pkg/store.py")
        store_cls = n(
            "pkg/store.py::VectorStore#class",
            Kind.CLASS,
            "VectorStore",
            "pkg/store.py",
            parent_id=store_file.id,
        )
        add = n("pkg/store.py::VectorStore.add#method", Kind.METHOD, "add", "pkg/store.py", parent_id=store_cls.id)
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        imp = r(f.id, "pkg.store", head="VectorStore", imported_name="VectorStore")
        call = r(caller.id, "store.add", head="store", ref_kind=RefKind.CALLS)
        index = RepoIndex([store_file, store_cls, add, f, caller], [imp, call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=add.id, confidence=Confidence.UNIQUE_NAME)

    def test_stoplist_does_not_apply_to_a_bare_single_segment_call(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        other = file_node("pkg/thing.py")
        get = n("pkg/thing.py::get#function", Kind.FUNCTION, "get", "pkg/thing.py", parent_id=other.id)
        call = r(caller.id, "get", head="get", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, caller, other, get], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.RESOLVED, target_id=get.id, confidence=Confidence.UNIQUE_NAME)


class TestModuleMemberAndWalkInternals:
    """Direct tests of the small internal helpers `_tier3`/`_resolve_type_name`
    lean on, for edge cases that are awkward to reach end-to-end through
    `resolve_repo_refs` (dangling IDs, an origin with no language family)."""

    def test_module_member_with_unknown_family_returns_an_empty_target(self):
        from indexter.index.resolve import _module_member, _module_member_strict

        index = RepoIndex([], [])

        assert _module_member(None, "x.py", "y", index) == Target()
        assert _module_member_strict(None, "x.py", "y", index) is None

    def test_tier3_bound_import_missing_from_outcomes_fails(self):
        from indexter.index.resolve import _tier3

        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        imp = r(caller.id, "pkg.auth", head="login", imported_name="login")
        index = RepoIndex([f, caller], [imp])

        result = _tier3(caller, "login", [], "login", "python", index, {}, {})

        assert result == Outcome(ResolveStatus.FAILED)

    def test_walk_from_module_target_short_circuits_on_external_mid_chain(self):
        from indexter.index.resolve import _walk_from_module_target

        index = RepoIndex([], [])
        outcome = Outcome(ResolveStatus.EXTERNAL, target_id=external_node_id("pkg"))

        assert _walk_from_module_target(outcome, "python", ["anything"], index, {}) == Outcome(
            ResolveStatus.EXTERNAL, target_id=external_node_id("pkg"), confidence=Confidence.IMPORTED
        )

    def test_walk_from_module_target_fails_on_a_non_resolved_start(self):
        from indexter.index.resolve import _walk_from_module_target

        index = RepoIndex([], [])

        assert _walk_from_module_target(Outcome(ResolveStatus.FAILED), "python", ["x"], index, {}) == Outcome(
            ResolveStatus.FAILED
        )
        assert _walk_from_module_target(Outcome(ResolveStatus.FAILED), "python", [], index, {}) == Outcome(
            ResolveStatus.FAILED
        )

    def test_walk_from_module_target_fails_on_a_dangling_target_id(self):
        from indexter.index.resolve import _walk_from_module_target

        index = RepoIndex([], [])
        outcome = Outcome(ResolveStatus.RESOLVED, target_id="ghost")

        assert _walk_from_module_target(outcome, "python", ["x"], index, {}) == Outcome(ResolveStatus.FAILED)

    def test_walk_from_module_target_fails_when_a_class_is_missing_the_member(self):
        from indexter.index.resolve import _walk_from_module_target

        f = file_node("pkg/a.py")
        cls = n("pkg/a.py::Foo#class", Kind.CLASS, "Foo", "pkg/a.py", parent_id=f.id)
        index = RepoIndex([f, cls], [])
        outcome = Outcome(ResolveStatus.RESOLVED, target_id=cls.id)

        assert _walk_from_module_target(outcome, "python", ["missing"], index, {}) == Outcome(ResolveStatus.FAILED)

    def test_call_ref_with_a_dangling_origin_fails(self):
        call = r("nonexistent-node-id", "helper", head="helper", ref_kind=RefKind.CALLS)
        index = RepoIndex([], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_call_ref_with_no_language_family_fails(self):
        f = file_node("styles.css", language="css")
        caller = n("styles.css::rule#function", Kind.FUNCTION, "rule", "styles.css", language="css", parent_id=f.id)
        call = r(caller.id, "helper", head="helper", ref_kind=RefKind.CALLS)
        index = RepoIndex([f, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)

    def test_call_ref_with_an_empty_chain_fails(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        call = r(caller.id, ".", head=None, ref_kind=RefKind.CALLS)
        index = RepoIndex([f, caller], [call])

        outcomes = resolve_repo_refs(index)

        assert outcomes[call.id] == Outcome(ResolveStatus.FAILED)


class TestForType:
    def test_for_type_that_is_only_separators_fails(self):
        impls_file = file_node("src/fmt.rs", language="rust")
        display_trait = n(
            "src/fmt.rs::Display#trait", Kind.TRAIT, "Display", "src/fmt.rs", language="rust", parent_id=impls_file.id
        )
        inherits = r(impls_file.id, "Display", head="Display", for_type="::", ref_kind=RefKind.INHERITS)
        index = RepoIndex([impls_file, display_trait], [inherits])

        outcomes = resolve_repo_refs(index)

        assert outcomes[inherits.id] == Outcome(ResolveStatus.FAILED)

    def test_for_type_resolving_to_a_non_type_kind_fails(self):
        impls_file = file_node("src/fmt.rs", language="rust")
        helper = n(
            "src/fmt.rs::Helper#function",
            Kind.FUNCTION,
            "Helper",
            "src/fmt.rs",
            language="rust",
            parent_id=impls_file.id,
        )
        inherits = r(impls_file.id, "Display", head="Display", for_type="Helper", ref_kind=RefKind.INHERITS)
        index = RepoIndex([impls_file, helper], [inherits])

        outcomes = resolve_repo_refs(index)

        assert outcomes[inherits.id] == Outcome(ResolveStatus.FAILED)

    def test_trait_implemented_for_a_struct_in_another_file(self):
        main_file = file_node("src/main.rs", language="rust")
        model = file_node("src/model.rs", language="rust")
        foo_struct = n(
            "src/model.rs::Foo#struct",
            Kind.STRUCT,
            "Foo",
            "src/model.rs",
            language="rust",
            parent_id=model.id,
        )
        decoy_file = file_node("vendor/model.rs", language="rust")
        decoy_foo = n(
            "vendor/model.rs::Foo#struct",
            Kind.STRUCT,
            "Foo",
            "vendor/model.rs",
            language="rust",
            parent_id=decoy_file.id,
        )
        fmt_file = file_node("src/fmt.rs", language="rust")
        traits_file = file_node("src/traits.rs", language="rust")
        display_trait = n(
            "src/traits.rs::Display#trait",
            Kind.TRAIT,
            "Display",
            "src/traits.rs",
            language="rust",
            parent_id=traits_file.id,
        )
        inherits = r(fmt_file.id, "Display", head="Display", for_type="crate::model::Foo", ref_kind=RefKind.INHERITS)
        index = RepoIndex(
            [main_file, model, foo_struct, decoy_file, decoy_foo, fmt_file, traits_file, display_trait], [inherits]
        )

        outcomes = resolve_repo_refs(index)

        outcome = outcomes[inherits.id]
        assert outcome.status == ResolveStatus.RESOLVED
        assert outcome.target_id == display_trait.id
        assert outcome.source_id == foo_struct.id

    def test_two_trait_impls_for_one_type_both_have_the_same_source(self):
        model = file_node("src/model.rs", language="rust")
        foo_struct = n(
            "src/model.rs::Foo#struct",
            Kind.STRUCT,
            "Foo",
            "src/model.rs",
            language="rust",
            parent_id=model.id,
        )
        traits_file = file_node("src/traits.rs", language="rust")
        display_trait = n(
            "src/traits.rs::Display#trait",
            Kind.TRAIT,
            "Display",
            "src/traits.rs",
            language="rust",
            parent_id=traits_file.id,
        )
        debug_trait = n(
            "src/traits.rs::Debug#trait",
            Kind.TRAIT,
            "Debug",
            "src/traits.rs",
            language="rust",
            parent_id=traits_file.id,
        )
        impls_file = file_node("src/fmt.rs", language="rust")
        display_impl = r(impls_file.id, "Display", head="Display", for_type="Foo", ref_kind=RefKind.INHERITS)
        debug_impl = r(impls_file.id, "Debug", head="Debug", for_type="Foo", ref_kind=RefKind.INHERITS)
        index = RepoIndex(
            [model, foo_struct, traits_file, display_trait, debug_trait, impls_file],
            [display_impl, debug_impl],
        )

        outcomes = resolve_repo_refs(index)

        assert outcomes[display_impl.id].target_id == display_trait.id
        assert outcomes[display_impl.id].source_id == foo_struct.id
        assert outcomes[debug_impl.id].target_id == debug_trait.id
        assert outcomes[debug_impl.id].source_id == foo_struct.id

    def test_unresolvable_for_type_fails_the_whole_reference(self):
        traits_file = file_node("src/traits.rs", language="rust")
        display_trait = n(
            "src/traits.rs::Display#trait",
            Kind.TRAIT,
            "Display",
            "src/traits.rs",
            language="rust",
            parent_id=traits_file.id,
        )
        impls_file = file_node("src/fmt.rs", language="rust")
        inherits = r(impls_file.id, "Display", head="Display", for_type="Nonexistent", ref_kind=RefKind.INHERITS)
        index = RepoIndex([traits_file, display_trait, impls_file], [inherits])

        outcomes = resolve_repo_refs(index)

        assert outcomes[inherits.id] == Outcome(ResolveStatus.FAILED)

    def test_for_type_ignored_when_the_head_itself_is_too_ambiguous(self):
        impls_file = file_node("src/fmt.rs", language="rust")
        model = file_node("src/model.rs", language="rust")
        foo_struct = n(
            "src/model.rs::Foo#struct",
            Kind.STRUCT,
            "Foo",
            "src/model.rs",
            language="rust",
            parent_id=model.id,
        )
        nodes: list[ResolveNode] = [impls_file, model, foo_struct]
        for i in range(8):
            fp = f"src/trait{i}.rs"
            file_n = file_node(fp, language="rust")
            trait = n(f"{fp}::Display#trait", Kind.TRAIT, "Display", fp, language="rust", parent_id=file_n.id)
            nodes += [file_n, trait]
        inherits = r(impls_file.id, "Display", head="Display", for_type="Foo", ref_kind=RefKind.INHERITS)
        index = RepoIndex(nodes, [inherits])

        outcomes = resolve_repo_refs(index)

        outcome = outcomes[inherits.id]
        assert outcome.status == ResolveStatus.TOO_AMBIGUOUS
        assert outcome.source_id is None


class TestDeterminismAndOrdering:
    def test_same_repository_contents_produce_the_same_outcome_regardless_of_input_order(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        other = file_node("pkg/thing.py")
        cls = n("pkg/thing.py::Thing#class", Kind.CLASS, "Thing", "pkg/thing.py", parent_id=other.id)
        normalize = n(
            "pkg/thing.py::Thing.normalize#method",
            Kind.METHOD,
            "normalize",
            "pkg/thing.py",
            parent_id=cls.id,
        )
        call = r(caller.id, "value.normalize", head="value", ref_kind=RefKind.CALLS)
        nodes = [f, caller, other, cls, normalize]

        first = resolve_repo_refs(RepoIndex(list(nodes), [call]))
        second = resolve_repo_refs(RepoIndex(list(reversed(nodes)), [call]))

        assert first[call.id] == second[call.id]

    def test_ambiguous_candidates_are_sorted(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        nodes: list[ResolveNode] = [f, caller]
        for i in range(3):
            fp = f"pkg/p{i}.py"
            file_n = file_node(fp)
            cls = n(f"{fp}::P#class", Kind.CLASS, "P", fp, parent_id=file_n.id)
            m = n(f"{fp}::P.parse#method", Kind.METHOD, "parse", fp, parent_id=cls.id)
            nodes += [file_n, cls, m]
        call = r(caller.id, "something.parse", head="something", ref_kind=RefKind.CALLS)
        index = RepoIndex(nodes, [call])

        outcomes = resolve_repo_refs(index)

        candidates = outcomes[call.id].candidates
        assert list(candidates) == sorted(candidates)

    def test_resolution_is_repeated_every_run_failed_becomes_resolved(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        call = r(caller.id, "normalize", head="normalize", ref_kind=RefKind.CALLS)
        before = resolve_repo_refs(RepoIndex([f, caller], [call]))
        assert before[call.id] == Outcome(ResolveStatus.FAILED)

        other = file_node("pkg/thing.py")
        normalize = n(
            "pkg/thing.py::normalize#function",
            Kind.FUNCTION,
            "normalize",
            "pkg/thing.py",
            parent_id=other.id,
        )
        after = resolve_repo_refs(RepoIndex([f, caller, other, normalize], [call]))

        assert after[call.id] == Outcome(
            ResolveStatus.RESOLVED, target_id=normalize.id, confidence=Confidence.UNIQUE_NAME
        )

    def test_unique_name_becomes_ambiguous_when_a_second_definition_appears(self):
        f = file_node("pkg/a.py")
        caller = n("pkg/a.py::main#function", Kind.FUNCTION, "main", "pkg/a.py", parent_id=f.id)
        other = file_node("pkg/thing.py")
        normalize = n(
            "pkg/thing.py::normalize#function",
            Kind.FUNCTION,
            "normalize",
            "pkg/thing.py",
            parent_id=other.id,
        )
        call = r(caller.id, "normalize", head="normalize", ref_kind=RefKind.CALLS)
        before = resolve_repo_refs(RepoIndex([f, caller, other, normalize], [call]))
        assert before[call.id] == Outcome(
            ResolveStatus.RESOLVED, target_id=normalize.id, confidence=Confidence.UNIQUE_NAME
        )

        second_file = file_node("pkg/other.py")
        second_normalize = n(
            "pkg/other.py::normalize#function", Kind.FUNCTION, "normalize", "pkg/other.py", parent_id=second_file.id
        )
        after = resolve_repo_refs(RepoIndex([f, caller, other, normalize, second_file, second_normalize], [call]))

        outcome = after[call.id]
        assert outcome.status == ResolveStatus.AMBIGUOUS
        assert sorted(outcome.candidates) == sorted([normalize.id, second_normalize.id])
