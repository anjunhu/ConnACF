"""
Unit tests for TopologyExtractor and BipartiteGraph.

Covers:
- record_mention adds edges correctly
- get_extracted_edges returns correct set
- get_bipartite_adjacency returns correct dict
- to_bipartite_graph returns correct BipartiteGraph
- BipartiteGraph.precision_recall_vs computes correct precision, recall, gs
- Empty graph edge cases
"""

import pytest

from connacf.attack.attackers.masleak.topology_extractor import (
    BipartiteGraph,
    TopologyExtractor,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_extractor() -> TopologyExtractor:
    return TopologyExtractor()


def make_graph(edges: list[tuple[int, int]]) -> BipartiteGraph:
    user_nodes = {u for u, _ in edges}
    item_nodes = {i for _, i in edges}
    return BipartiteGraph(user_nodes=user_nodes, item_nodes=item_nodes, edges=set(edges))


# ---------------------------------------------------------------------------
# TopologyExtractor — record_mention
# ---------------------------------------------------------------------------


class TestRecordMention:
    def test_single_mention_adds_edge(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        assert (1, 10) in te.get_extracted_edges()

    def test_duplicate_mention_does_not_duplicate_edge(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        te.record_mention(1, 10, turn=1)
        assert te.get_extracted_edges() == {(1, 10)}

    def test_multiple_users_multiple_items(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        te.record_mention(1, 20, turn=0)
        te.record_mention(2, 10, turn=1)
        assert te.get_extracted_edges() == {(1, 10), (1, 20), (2, 10)}

    def test_turn_does_not_affect_edge_set(self):
        te = make_extractor()
        te.record_mention(3, 30, turn=99)
        assert (3, 30) in te.get_extracted_edges()


# ---------------------------------------------------------------------------
# TopologyExtractor — get_extracted_edges
# ---------------------------------------------------------------------------


class TestGetExtractedEdges:
    def test_empty_extractor_returns_empty_set(self):
        te = make_extractor()
        assert te.get_extracted_edges() == set()

    def test_returns_copy_not_internal_reference(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        edges = te.get_extracted_edges()
        edges.add((99, 99))  # mutate the returned set
        # internal state should be unchanged
        assert (99, 99) not in te.get_extracted_edges()

    def test_correct_edges_after_multiple_records(self):
        te = make_extractor()
        expected = {(1, 10), (2, 20), (3, 30)}
        for u, i in expected:
            te.record_mention(u, i, turn=0)
        assert te.get_extracted_edges() == expected


# ---------------------------------------------------------------------------
# TopologyExtractor — get_bipartite_adjacency
# ---------------------------------------------------------------------------


class TestGetBipartiteAdjacency:
    def test_empty_extractor_returns_empty_dict(self):
        te = make_extractor()
        assert te.get_bipartite_adjacency() == {}

    def test_single_user_multiple_items(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        te.record_mention(1, 20, turn=1)
        te.record_mention(1, 30, turn=2)
        adj = te.get_bipartite_adjacency()
        assert adj == {1: {10, 20, 30}}

    def test_multiple_users(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        te.record_mention(2, 20, turn=0)
        adj = te.get_bipartite_adjacency()
        assert adj[1] == {10}
        assert adj[2] == {20}

    def test_returns_copy_not_internal_reference(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        adj = te.get_bipartite_adjacency()
        adj[1].add(999)  # mutate returned dict
        assert 999 not in te.get_bipartite_adjacency()[1]

    def test_shared_item_across_users(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        te.record_mention(2, 10, turn=0)
        adj = te.get_bipartite_adjacency()
        assert 10 in adj[1]
        assert 10 in adj[2]


# ---------------------------------------------------------------------------
# TopologyExtractor — to_bipartite_graph
# ---------------------------------------------------------------------------


class TestToBipartiteGraph:
    def test_empty_extractor_returns_empty_graph(self):
        te = make_extractor()
        g = te.to_bipartite_graph()
        assert g.edges == set()
        assert g.user_nodes == set()
        assert g.item_nodes == set()

    def test_user_and_item_nodes_populated(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        te.record_mention(2, 20, turn=0)
        g = te.to_bipartite_graph()
        assert g.user_nodes == {1, 2}
        assert g.item_nodes == {10, 20}

    def test_edges_match_recorded_mentions(self):
        te = make_extractor()
        te.record_mention(1, 10, turn=0)
        te.record_mention(1, 20, turn=1)
        te.record_mention(2, 10, turn=0)
        g = te.to_bipartite_graph()
        assert g.edges == {(1, 10), (1, 20), (2, 10)}

    def test_returns_bipartite_graph_instance(self):
        te = make_extractor()
        g = te.to_bipartite_graph()
        assert isinstance(g, BipartiteGraph)


# ---------------------------------------------------------------------------
# BipartiteGraph — precision_recall_vs
# ---------------------------------------------------------------------------


class TestPrecisionRecallVs:
    def test_perfect_match(self):
        g = make_graph([(1, 10), (2, 20)])
        gt = make_graph([(1, 10), (2, 20)])
        result = g.precision_recall_vs(gt)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(1.0)
        assert result["gs"] == pytest.approx(1.0)

    def test_no_overlap(self):
        g = make_graph([(1, 10)])
        gt = make_graph([(2, 20)])
        result = g.precision_recall_vs(gt)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["gs"] == pytest.approx(0.0)

    def test_partial_overlap_precision(self):
        # extracted has 2 edges, 1 correct → precision = 0.5
        g = make_graph([(1, 10), (1, 99)])
        gt = make_graph([(1, 10)])
        result = g.precision_recall_vs(gt)
        assert result["precision"] == pytest.approx(0.5)
        assert result["recall"] == pytest.approx(1.0)

    def test_partial_overlap_recall(self):
        # extracted has 1 of 2 ground-truth edges → recall = 0.5
        g = make_graph([(1, 10)])
        gt = make_graph([(1, 10), (2, 20)])
        result = g.precision_recall_vs(gt)
        assert result["precision"] == pytest.approx(1.0)
        assert result["recall"] == pytest.approx(0.5)

    def test_gs_is_f1_of_precision_and_recall(self):
        g = make_graph([(1, 10), (1, 99)])
        gt = make_graph([(1, 10), (2, 20)])
        result = g.precision_recall_vs(gt)
        p, r = result["precision"], result["recall"]
        expected_gs = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        assert result["gs"] == pytest.approx(expected_gs)

    def test_empty_extracted_vs_nonempty_gt(self):
        g = make_graph([])
        gt = make_graph([(1, 10)])
        result = g.precision_recall_vs(gt)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["gs"] == pytest.approx(0.0)

    def test_nonempty_extracted_vs_empty_gt(self):
        g = make_graph([(1, 10)])
        gt = make_graph([])
        result = g.precision_recall_vs(gt)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)  # 0 / (0+0) → 0.0
        assert result["gs"] == pytest.approx(0.0)

    def test_both_empty(self):
        g = make_graph([])
        gt = make_graph([])
        result = g.precision_recall_vs(gt)
        assert result["precision"] == pytest.approx(0.0)
        assert result["recall"] == pytest.approx(0.0)
        assert result["gs"] == pytest.approx(0.0)
