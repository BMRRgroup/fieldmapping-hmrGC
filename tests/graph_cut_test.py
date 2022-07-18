import pytest
import numpy as np
from hmrGC.constants import INFTY
from hmrGC.graph_cut import GraphCut


@pytest.fixture
def nodes_arrays():
    """
    Modified toy example from Shah et al.: Optimal surface segmentation
    with convex priors in irregularly sampled space, DOI: 10.1016/j.media.2019.02.004
    """
    nodes_arrays = {}
    mask = np.ones((4, 1, 1))
    mask[-2] = 0
    intra_column_mask = np.ones((3, 6))
    intra_column_mask[-1, 1:] = 0
    intra_column_cost = intra_column_mask.copy()
    inter_column_cost = intra_column_mask.copy()
    intra_column_mask = intra_column_mask.astype(np.bool_)
    inter_column_cost[0] = [4, 16, 21, 25, 34, 52]
    inter_column_cost[1] = [1, 3, 12, 28, 37, 50]
    inter_column_cost[-1, 0] = 4
    nodes_arrays['voxel_mask'] = mask
    nodes_arrays['inter_column_cost'] = inter_column_cost
    nodes_arrays['intra_column_cost'] = intra_column_cost
    nodes_arrays['intra_column_mask'] = intra_column_mask
    nodes_arrays['voxel_weighting'] = np.ones_like(intra_column_mask[:, 0])
    return nodes_arrays


def test_graphcut_init(nodes_arrays, test_gpu):
    if test_gpu:
        g = GraphCut(nodes_arrays)
        assert g.nodes_id.shape == nodes_arrays['intra_column_mask'].shape
        np.testing.assert_equal(g.nodes_id[0], np.arange(6))


def test_graphcut_set_edges_and_tedges(nodes_arrays, test_gpu):
    if test_gpu:
        g = GraphCut(nodes_arrays)
        g.set_edges_and_tedges()
        assert g.edges.shape == (37, 4)
        assert g.tedges.shape == (7, 3)


def test_graphcut_mincut(nodes_arrays):
    g = GraphCut(nodes_arrays)
    g.set_edges_and_tedges()
    g.mincut()
    assert g.maxflow == 40
    np.testing.assert_equal(g.mincut, [False, True, True, True, True, True, False,
                                       False, True, True, True, True, False])


def test_graphcut_get_map(nodes_arrays, test_gpu):
    if test_gpu:
        g = GraphCut(nodes_arrays)
        g.set_edges_and_tedges()
        g.mincut()
        map = g.get_map()
        assert map.shape == (3,)
        np.testing.assert_equal(map, [0, 1, 0])


def test_graphcut_get_intra_edges_weights(nodes_arrays, test_gpu):
    if test_gpu:
        g = GraphCut(nodes_arrays)
        edges, tedges = g.get_intra_edges_weights()
        assert edges.shape == (10, 4)
        assert tedges.shape == (6, 3)
        num_nodes = np.max(g.nodes_id)+1
        assert np.max(edges[:, 0]) < num_nodes
        assert np.max(tedges[:, 0]) < num_nodes
        assert np.max(edges[:, 1]) < num_nodes
        np.testing.assert_equal(edges[:, 2], 10*np.ones_like(edges[:, 2]))
        assert np.max(edges) <= INFTY
        assert np.max(tedges) <= INFTY


def test_graphcut_set_coordinate_information(nodes_arrays, test_gpu):
    if test_gpu:
        g = GraphCut(nodes_arrays)
        g.set_coordinate_information()
        np.testing.assert_equal(g.voxel_id, [0, 1, 3])
        np.testing.assert_equal(g.voxel_id_rev[g.voxel_id], np.arange(3))
        np.testing.assert_equal(g.coordinates, np.array([[0, 0, 0], [1, 0, 0],
                                                         [3, 0, 0]]))


def test_graphcut_get_inter_edges_weights(nodes_arrays, test_gpu):
    if test_gpu:
        g = GraphCut(nodes_arrays)
        g.inter_column_scaling = 1e0
        g.set_coordinate_information()
        edges, tedges = g.get_inter_edges_weights(0)
        assert edges.shape == (27, 4)
        assert tedges.shape == (1, 3)
        edges, tedges = g.get_inter_edges_weights(0, metric='abs')
        # Values should match Table 1, Shah et al.: Optimal surface segmentation
        # with convex priors in irregularly sampled space
        np.testing.assert_equal(edges, np.array([[0, 7, 2, 0],
                                                 [0, 8, 1, 0],
                                                 [1, 8, 8, 8],
                                                 [1, 9, 4, 4],
                                                 [2, 9, 5, 5],
                                                 [3, 9, 4, 4],
                                                 [4, 9, 3, 3],
                                                 [4, 10, 6, 6],
                                                 [5, 10, 3, 3],
                                                 [5, 11, 13, 13]]))
        np.testing.assert_equal(tedges, [[5, 0, 2]])
        edges, tedges = g.get_inter_edges_weights(1)
        np.testing.assert_equal(edges, np.zeros((0, 4)))
        np.testing.assert_equal(tedges, np.zeros((0, 3)))
