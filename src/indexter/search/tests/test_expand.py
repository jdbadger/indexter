from indexter.search.expand import expand, hit_context
from indexter.search.tests.conftest import insert_edge, insert_node
from indexter.search.types import Seed


class TestHitContext:
    def test_no_callers_or_callees_or_container(self, graph_conn):
        insert_node(graph_conn, id="a")
        context = hit_context(graph_conn, "a")
        assert context.callers == ()
        assert context.caller_total == 0
        assert context.callees == ()
        assert context.callee_total == 0
        assert context.container is None

    def test_callers_ordered_by_confidence_then_id(self, graph_conn):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="ambiguous_caller")
        insert_node(graph_conn, id="exact_caller")
        insert_node(graph_conn, id="unique_name_caller")
        insert_edge(graph_conn, source="ambiguous_caller", target="target", confidence="ambiguous")
        insert_edge(graph_conn, source="exact_caller", target="target", confidence="exact")
        insert_edge(graph_conn, source="unique_name_caller", target="target", confidence="unique_name")

        context = hit_context(graph_conn, "target")

        assert [c.node_id for c in context.callers] == ["exact_caller", "unique_name_caller", "ambiguous_caller"]
        assert context.caller_total == 3

    def test_callees_ordered_and_capped_with_total(self, graph_conn):
        insert_node(graph_conn, id="source")
        for i in range(5):
            insert_node(graph_conn, id=f"callee{i}")
            insert_edge(graph_conn, source="source", target=f"callee{i}", confidence="exact")

        context = hit_context(graph_conn, "source")

        assert [c.node_id for c in context.callees] == ["callee0", "callee1", "callee2"]
        assert context.callee_total == 5

    def test_repeated_calls_to_same_target_count_as_one_caller(self, graph_conn):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="caller")
        insert_edge(graph_conn, source="caller", target="target", confidence="exact", line=1)
        insert_edge(graph_conn, source="caller", target="target", confidence="exact", line=2)

        context = hit_context(graph_conn, "target")

        assert [c.node_id for c in context.callers] == ["caller"]
        assert context.caller_total == 1

    def test_best_confidence_kept_when_same_source_has_multiple_edges(self, graph_conn):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="caller")
        insert_edge(graph_conn, source="caller", target="target", confidence="ambiguous", line=1)
        insert_edge(graph_conn, source="caller", target="target", confidence="exact", line=2)

        context = hit_context(graph_conn, "target")

        assert context.callers[0].confidence == "exact"

    def test_ties_broken_by_id(self, graph_conn):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="b_caller")
        insert_node(graph_conn, id="a_caller")
        insert_edge(graph_conn, source="b_caller", target="target", confidence="exact")
        insert_edge(graph_conn, source="a_caller", target="target", confidence="exact")

        context = hit_context(graph_conn, "target")

        assert [c.node_id for c in context.callers] == ["a_caller", "b_caller"]

    def test_only_calls_edges_counted_not_imports_or_inherits(self, graph_conn):
        insert_node(graph_conn, id="target")
        insert_node(graph_conn, id="importer")
        insert_edge(graph_conn, source="importer", target="target", kind="imports", confidence="exact")

        context = hit_context(graph_conn, "target")

        assert context.callers == ()
        assert context.caller_total == 0

    def test_class_like_parent_is_the_container(self, graph_conn):
        insert_node(graph_conn, id="Store", kind="class")
        insert_node(graph_conn, id="Store.add", kind="method", parent_id="Store")

        context = hit_context(graph_conn, "Store.add")

        assert context.container is not None
        assert context.container.node_id == "Store"
        assert context.container.qualified_name == "Store"

    def test_non_class_like_parent_is_not_a_container(self, graph_conn):
        insert_node(graph_conn, id="a.py", kind="file")
        insert_node(graph_conn, id="helper", kind="function", parent_id="a.py")

        context = hit_context(graph_conn, "helper")

        assert context.container is None

    def test_no_parent_means_no_container(self, graph_conn):
        insert_node(graph_conn, id="a")
        context = hit_context(graph_conn, "a")
        assert context.container is None


def _seed(node_id, *, qualified_name=None, weight=1.0):
    return Seed(node_id=node_id, qualified_name=qualified_name or node_id, weight=weight)


class TestExpand:
    def test_no_seeds_returns_empty(self, graph_conn):
        assert expand(graph_conn, [], frozenset()) == ()

    def test_callee_of_a_hit_is_related(self, graph_conn):
        insert_node(graph_conn, id="hit", qualified_name="Hit")
        insert_node(graph_conn, id="callee", qualified_name="Callee")
        insert_edge(graph_conn, source="hit", target="callee", kind="calls", confidence="exact")

        related = expand(graph_conn, [_seed("hit", qualified_name="Hit")], frozenset())

        assert len(related) == 1
        assert related[0].node_id == "callee"
        assert related[0].reason == "called by Hit, which matched"

    def test_caller_of_a_hit_is_related(self, graph_conn):
        insert_node(graph_conn, id="hit", qualified_name="Hit")
        insert_node(graph_conn, id="caller", qualified_name="Caller")
        insert_edge(graph_conn, source="caller", target="hit", kind="calls", confidence="exact")

        related = expand(graph_conn, [_seed("hit", qualified_name="Hit")], frozenset())

        assert len(related) == 1
        assert related[0].node_id == "caller"
        assert related[0].reason == "calls Hit, which matched"

    def test_hub_damping(self, graph_conn):
        insert_node(graph_conn, id="hit")
        insert_node(graph_conn, id="hub", degree=41)
        insert_edge(graph_conn, source="hit", target="hub", kind="calls", confidence="exact")

        related = expand(graph_conn, [_seed("hit")], frozenset())

        assert related == ()

    def test_already_shown_node_is_excluded(self, graph_conn):
        insert_node(graph_conn, id="hit")
        insert_node(graph_conn, id="other_hit")
        insert_edge(graph_conn, source="hit", target="other_hit", kind="calls", confidence="exact")

        related = expand(graph_conn, [_seed("hit")], frozenset({"other_hit"}))

        assert related == ()

    def test_external_module_neighbor_is_excluded(self, graph_conn):
        insert_node(graph_conn, id="hit")
        insert_node(graph_conn, id="external::pkg", kind="external_module")
        insert_edge(graph_conn, source="hit", target="external::pkg", kind="imports", confidence="imported")

        related = expand(graph_conn, [_seed("hit")], frozenset())

        assert related == ()

    def test_class_member_is_related_with_membership_reason(self, graph_conn):
        insert_node(graph_conn, id="Store", kind="class", qualified_name="Store")
        insert_node(graph_conn, id="Store.add", kind="method", parent_id="Store", qualified_name="Store.add")
        insert_edge(graph_conn, source="Store", target="Store.add", kind="contains", confidence="exact")

        related = expand(graph_conn, [_seed("Store", qualified_name="Store")], frozenset({"Store"}))

        assert len(related) == 1
        assert related[0].node_id == "Store.add"
        assert related[0].reason == "member of Store, which matched"

    def test_container_of_a_member_seed_is_related_with_contains_reason(self, graph_conn):
        insert_node(graph_conn, id="Store", kind="class", qualified_name="Store")
        insert_node(graph_conn, id="Store.add", kind="method", parent_id="Store", qualified_name="Store.add")
        insert_edge(graph_conn, source="Store", target="Store.add", kind="contains", confidence="exact")

        related = expand(graph_conn, [_seed("Store.add", qualified_name="Store.add")], frozenset({"Store.add"}))

        assert len(related) == 1
        assert related[0].node_id == "Store"
        assert related[0].reason == "contains Store.add, which matched"

    def test_contains_edge_is_not_followed_when_seed_is_a_file(self, graph_conn):
        insert_node(graph_conn, id="a.py", kind="file")
        insert_node(graph_conn, id="helper", kind="function", parent_id="a.py")
        insert_edge(graph_conn, source="a.py", target="helper", kind="contains", confidence="exact")

        related = expand(graph_conn, [_seed("a.py")], frozenset())

        assert related == ()

    def test_contains_edge_is_not_followed_when_neighbor_is_a_file(self, graph_conn):
        insert_node(graph_conn, id="a.py", kind="file")
        insert_node(graph_conn, id="helper", kind="function", parent_id="a.py")
        insert_edge(graph_conn, source="a.py", target="helper", kind="contains", confidence="exact")

        related = expand(graph_conn, [_seed("helper")], frozenset())

        assert related == ()

    def test_inheritance_base_class_direction(self, graph_conn):
        insert_node(graph_conn, id="Sub", qualified_name="Sub")
        insert_node(graph_conn, id="Base", qualified_name="Base")
        insert_edge(graph_conn, source="Sub", target="Base", kind="inherits", confidence="exact")

        related = expand(graph_conn, [_seed("Sub", qualified_name="Sub")], frozenset())

        assert related[0].node_id == "Base"
        assert related[0].reason == "base class of Sub, which matched"

    def test_inheritance_subclass_direction(self, graph_conn):
        insert_node(graph_conn, id="Base", qualified_name="Base")
        insert_node(graph_conn, id="Sub", qualified_name="Sub")
        insert_edge(graph_conn, source="Sub", target="Base", kind="inherits", confidence="exact")

        related = expand(graph_conn, [_seed("Base", qualified_name="Base")], frozenset())

        assert related[0].node_id == "Sub"
        assert related[0].reason == "subclass of Base, which matched"

    def test_imports_both_directions(self, graph_conn):
        insert_node(graph_conn, id="importer", qualified_name="importer")
        insert_node(graph_conn, id="imported", qualified_name="imported")
        insert_edge(graph_conn, source="importer", target="imported", kind="imports", confidence="exact")

        from_importer = expand(graph_conn, [_seed("importer", qualified_name="importer")], frozenset())
        from_imported = expand(graph_conn, [_seed("imported", qualified_name="imported")], frozenset())

        assert from_importer[0].node_id == "imported"
        assert from_importer[0].reason == "imported by importer, which matched"
        assert from_imported[0].node_id == "importer"
        assert from_imported[0].reason == "imports imported, which matched"

    def test_ambiguous_alone_is_dropped(self, graph_conn):
        insert_node(graph_conn, id="hit")
        insert_node(graph_conn, id="maybe")
        insert_edge(graph_conn, source="hit", target="maybe", kind="calls", confidence="ambiguous")

        related = expand(graph_conn, [_seed("hit")], frozenset())

        assert related == ()

    def test_ambiguous_adds_weight_to_a_confident_path(self, graph_conn):
        insert_node(graph_conn, id="x", qualified_name="X")
        insert_node(graph_conn, id="y", qualified_name="Y")
        insert_node(graph_conn, id="seed1")
        insert_node(graph_conn, id="seed2")
        insert_edge(graph_conn, source="seed1", target="x", kind="calls", confidence="ambiguous")
        insert_edge(graph_conn, source="seed2", target="x", kind="calls", confidence="exact")
        insert_edge(graph_conn, source="seed2", target="y", kind="calls", confidence="exact")

        seeds = [_seed("seed1", weight=0.5), _seed("seed2", weight=0.3)]
        related = expand(graph_conn, seeds, frozenset())

        assert [r.node_id for r in related] == ["x", "y"]

    def test_reached_from_more_hits_ranks_higher(self, graph_conn):
        insert_node(graph_conn, id="x")
        insert_node(graph_conn, id="y")
        insert_node(graph_conn, id="seed1")
        insert_node(graph_conn, id="seed3")
        insert_edge(graph_conn, source="seed1", target="x", kind="calls", confidence="exact")
        insert_edge(graph_conn, source="seed3", target="x", kind="calls", confidence="exact")
        insert_edge(graph_conn, source="seed1", target="y", kind="calls", confidence="exact")

        seeds = [_seed("seed1", weight=1.0), _seed("seed3", weight=0.5)]
        related = expand(graph_conn, seeds, frozenset())

        assert [r.node_id for r in related] == ["x", "y"]

    def test_at_most_five_related(self, graph_conn):
        insert_node(graph_conn, id="hit")
        for i in range(12):
            insert_node(graph_conn, id=f"n{i:02d}")
            insert_edge(graph_conn, source="hit", target=f"n{i:02d}", kind="calls", confidence="exact")

        related = expand(graph_conn, [_seed("hit")], frozenset())

        assert [r.node_id for r in related] == ["n00", "n01", "n02", "n03", "n04"]

    def test_several_seeds_note_how_many_others_contributed(self, graph_conn):
        insert_node(graph_conn, id="Base", qualified_name="Base")
        insert_node(graph_conn, id="seed1", qualified_name="Seed1")
        insert_node(graph_conn, id="seed4", qualified_name="Seed4")
        insert_edge(graph_conn, source="seed1", target="Base", kind="inherits", confidence="exact")
        insert_edge(graph_conn, source="seed4", target="Base", kind="inherits", confidence="exact")

        seeds = [_seed("seed1", qualified_name="Seed1", weight=1.0), _seed("seed4", qualified_name="Seed4", weight=0.5)]
        related = expand(graph_conn, seeds, frozenset())

        assert related[0].reason == "base class of Seed1, which matched (+1 more)"

    def test_strongest_contribution_tie_broken_by_edge_kind_order(self, graph_conn):
        insert_node(graph_conn, id="hit", qualified_name="Hit")
        insert_node(graph_conn, id="neighbor")
        insert_edge(graph_conn, source="neighbor", target="hit", kind="inherits", confidence="exact")
        insert_edge(graph_conn, source="hit", target="neighbor", kind="calls", confidence="exact")

        related = expand(graph_conn, [_seed("hit", qualified_name="Hit")], frozenset())

        assert related[0].reason == "called by Hit, which matched"


class TestExpandIntegration:
    """One check over the M4-style synced fixture repo (`conn`), rather
    than a hand-built graph -- design.md decision 10."""

    def test_class_seed_relates_its_own_method(self, conn):
        seed = Seed(node_id="src/walker.py::Store#class", qualified_name="Store", weight=1 / 61)

        related = expand(conn, [seed], frozenset({seed.node_id}))

        assert len(related) == 1
        assert related[0].node_id == "src/walker.py::Store.add#method"
        assert related[0].reason == "member of Store, which matched"

    def test_file_seed_does_not_relate_its_contents_or_imports(self, conn):
        seed = Seed(node_id="src/walker.py::#file", qualified_name="src/walker.py", weight=1 / 61)

        related = expand(conn, [seed], frozenset({seed.node_id}))

        assert related == ()


class TestExpandEdgeCases:
    def test_seed_with_no_edges_returns_empty(self, graph_conn):
        insert_node(graph_conn, id="lonely")

        related = expand(graph_conn, [_seed("lonely")], frozenset())

        assert related == ()

    def test_self_referential_edge_is_not_its_own_related_node(self, graph_conn):
        insert_node(graph_conn, id="recursive")
        insert_edge(graph_conn, source="recursive", target="recursive", kind="calls", confidence="exact")

        related = expand(graph_conn, [_seed("recursive")], frozenset())

        assert related == ()

    def test_edge_to_a_node_missing_from_the_nodes_table_is_skipped(self, graph_conn):
        insert_node(graph_conn, id="hit")
        insert_edge(graph_conn, source="hit", target="gone", kind="calls", confidence="exact")

        related = expand(graph_conn, [_seed("hit")], frozenset())

        assert related == ()
