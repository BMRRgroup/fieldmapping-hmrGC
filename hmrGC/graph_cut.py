import numpy as np
import maxflow
import os
from hmrGC.helper import ChunkArr, get_arr_chunks, xp_function, \
    free_mempool, move2cpu, move2gpu, CUDA_OutOfMemory
from hmrGC.constants import INFTY


class GraphCut():
    """
    3D surface segmentation using the minimum cut of a graph

    Shah et al.: Optimal surface segmentation with convex priors in irregularly
    sampled space, Medical Image Analysis, DOI: 10.1016/j.media.2019.02.004

    Boykov et al.: An Experimental Comparison of Min-Cut/Max-Flow Algorithms for
    Energy Minimization in Vision., IEEE Transactions on Pattern Analysis and
    Machine Intelligence, DOI: 10.1109/TPAMI.2004.60

    The python package 'PyMaxflow' is used for graph construction and solving.
    (https://github.com/pmneila/PyMaxflow)
    """

    def __init__(self, nodes_arrays):
        """Initialize GraphCut class

        :param nodes_arrays: dict with keys 'intra_column_mask', 'intra_column_cost',
                             'inter_column_cost', 'voxel_mask'

        """
        self.intra_column_mask = nodes_arrays['intra_column_mask']
        self.intra_column_cost = nodes_arrays['intra_column_cost']
        self.inter_column_cost = nodes_arrays['inter_column_cost']
        self.voxel_num = nodes_arrays['voxel_mask'].shape
        self.voxel_mask_reshaped = np.reshape(nodes_arrays['voxel_mask'],
                                              int(np.prod(self.voxel_num)))
        nodes_id = np.zeros_like(self.intra_column_mask, dtype=np.uint32)
        nodes_id[self.intra_column_mask] = np.arange(np.sum(self.intra_column_mask))
        self.nodes_id = nodes_id
        self.nodes_per_voxel = np.sum(self.intra_column_mask, axis=-1)

        self.isotropic_scaling = [1, 1, 1]
        self.intra_column_scaling = 1e1
        self.inter_column_scaling = 1e1
        if 'voxel_weighting' in nodes_arrays:
            self.voxel_weighting = nodes_arrays['voxel_weighting']
            self.voxel_weighting_intra_column = True
            self.voxel_weighting_inter_column = True
        else:
            self.voxel_weighting_intra_column = False
            self.voxel_weighting_inter_column = False

        self.chunk_mode = False
        if os.environ.get('BMRR_USE_GPU') == '1':
            self.use_gpu = True
        else:
            self.use_gpu = False

    @xp_function
    def set_edges_and_tedges(self, xp=np):
        """Set edges and terminal edges.

        See :meth:`pygandalf.GraphCut.get_intra_edges_weights` and
        :meth:`pygandalf.GraphCut.get_inter_edges_weights`

        """
        intra_edges, intra_tedges = self.get_intra_edges_weights()
        self.set_coordinate_information()
        inter_edges_x, inter_tedges_x = self.get_inter_edges_weights(dim=0)
        inter_edges_y, inter_tedges_y = self.get_inter_edges_weights(dim=1)
        inter_edges_z, inter_tedges_z = self.get_inter_edges_weights(dim=2)

        if not self.chunk_mode:
            try:
                edges = move2cpu(xp.concatenate((move2gpu(intra_edges, xp),
                                                 move2gpu(inter_edges_x, xp),
                                                 move2gpu(inter_edges_y, xp),
                                                 move2gpu(inter_edges_z, xp)),
                                                axis=0), xp)
                tedges = move2cpu(xp.concatenate((move2gpu(intra_tedges, xp),
                                                  move2gpu(inter_tedges_x, xp),
                                                  move2gpu(inter_tedges_y, xp),
                                                  move2gpu(inter_tedges_z, xp)),
                                                 axis=0), xp)
            except CUDA_OutOfMemory:
                self.chunk_mode = True
        if self.chunk_mode:
            edges = np.concatenate((intra_edges, inter_edges_x, inter_edges_y,
                                    inter_edges_z), axis=0)
            tedges = np.concatenate((intra_tedges, inter_tedges_x,
                                     inter_tedges_y, inter_tedges_z), axis=0)
        self.edges = edges
        self.tedges = tedges
        free_mempool()

    def mincut(self):
        edges = self.edges
        tedges = self.tedges
        num_nodes = np.max(self.nodes_id)+1

        g = maxflow.Graph[int](num_nodes, edges.shape[0])
        g.add_nodes(num_nodes)
        g.add_edges(edges[:, 0], edges[:, 1], edges[:, 2], edges[:, 3])
        g.add_grid_tedges(tedges[:, 0], tedges[:, 1], tedges[:, 2])

        self.maxflow = g.maxflow()
        self.mincut = g.get_grid_segments(np.arange(num_nodes))

    @xp_function
    def get_map(self, xp=np):
        mincut = move2gpu(self.mincut, xp)
        nodes_id = move2gpu(self.nodes_id, xp)
        intra_column_mask = move2gpu(self.intra_column_mask, xp)

        # Find the mincut-minimum per voxel
        mask_mincut = intra_column_mask.copy()
        mask_mincut[intra_column_mask] = mincut
        nodes_mincut = nodes_id
        nodes_mincut[mask_mincut] = 0
        map_masked = xp.argmax(nodes_mincut, axis=-1)
        return move2cpu(map_masked, xp)

    @xp_function
    def get_intra_edges_weights(self, xp=np):
        nodes_per_voxel = move2gpu(self.nodes_per_voxel, xp)
        nodes_id = move2gpu(self.nodes_id, xp)
        intra_column_cost = move2gpu(self.intra_column_cost, xp)
        intra_column_mask = move2gpu(self.intra_column_mask, xp)

        if self.voxel_weighting_intra_column:
            voxel_weighting = move2gpu(self.voxel_weighting, xp)
            intra_column_cost = intra_column_cost.T * voxel_weighting
            intra_column_cost = intra_column_cost.T

        chunksize = nodes_per_voxel.shape[0]
        reduce_gpu_mem = False
        while True:
            try:
                num_chunks = int(np.ceil(nodes_per_voxel.shape[0]/chunksize))
                edges = ChunkArr('uint32', reduce_gpu_mem, shape=(0, 4), xp=xp)
                tedges = ChunkArr('uint32', reduce_gpu_mem, shape=(0, 3), xp=xp)

                for i in range(num_chunks):
                    edges, tedges = \
                        self.__chunk_func_get_intra_edges_weights(chunksize, i,
                                                                  num_chunks,
                                                                  nodes_per_voxel,
                                                                  nodes_id,
                                                                  intra_column_cost,
                                                                  intra_column_mask,
                                                                  edges, tedges, xp)
                return edges.to_numpy(), tedges.to_numpy()
            except CUDA_OutOfMemory:
                reduce_gpu_mem = True
                chunksize = chunksize // 2

    def __chunk_func_get_intra_edges_weights(self, chunksize, i, num_chunks,
                                             nodes_per_voxel, nodes_id,
                                             intra_column_cost, intra_column_mask,
                                             edges, tedges, xp):
        nodes_id_chunk = get_arr_chunks(chunksize, nodes_id, i, num_chunks)
        intra_column_cost_chunk = get_arr_chunks(chunksize, intra_column_cost,
                                                 i, num_chunks)
        intra_column_mask_chunk = get_arr_chunks(chunksize, intra_column_mask,
                                                 i, num_chunks)
        nodes_per_voxel_chunk = get_arr_chunks(chunksize, nodes_per_voxel, i,
                                               num_chunks)

        helper_mask = xp.arange(len(nodes_per_voxel_chunk))
        # Calculate edges connected to source or sink
        tedges1 = xp.zeros((len(helper_mask), 3), dtype=xp.uint32)
        tedges1[:, 0] = nodes_id_chunk[helper_mask, nodes_per_voxel_chunk-1]
        tedges1[:, 2] = xp.around(self.intra_column_scaling *
                                  intra_column_cost_chunk[helper_mask,
                                                          nodes_per_voxel_chunk-1])
        tedges2 = xp.zeros((len(helper_mask), 3), dtype=xp.uint32)
        tedges2[:, 0] = nodes_id_chunk[helper_mask, 0]
        tedges2[:, 1] = INFTY
        tedges1 = xp.concatenate((tedges1, tedges2))
        del tedges2
        # Calculate edges between nodes
        helper_mask2 = intra_column_mask_chunk.copy()
        helper_mask2[helper_mask, nodes_per_voxel_chunk-1] = False
        del helper_mask
        edges1 = xp.zeros((int(xp.sum(helper_mask2)), 4), dtype=xp.uint32)
        edges1[:, 0] = nodes_id_chunk[helper_mask2]
        edges1[:, 1] = edges1[:, 0] + 1
        edges1[:, 2] = xp.around(self.intra_column_scaling *
                                 intra_column_cost_chunk[helper_mask2])
        edges1[:, 3] = INFTY

        edges.concatenate(edges1)
        tedges.concatenate(tedges1)
        return edges, tedges

    @xp_function
    def set_coordinate_information(self, xp=np):
        mask = move2gpu(self.voxel_mask_reshaped, xp)
        voxel_id = xp.squeeze(xp.argwhere(mask != 0))
        voxel_id_rev = xp.ones(int(np.prod(self.voxel_num)), dtype=xp.uint32)*len(voxel_id)
        voxel_id_rev[voxel_id] = xp.arange(len(voxel_id), dtype=xp.uint32)
        X, Y, Z = xp.unravel_index(xp.arange(int(np.prod(self.voxel_num))),
                                   self.voxel_num)
        coordinates = xp.array([X, Y, Z]).transpose()[voxel_id]

        self.voxel_id = move2cpu(voxel_id, xp)
        self.voxel_id_rev = move2cpu(voxel_id_rev, xp)
        self.coordinates = move2cpu(coordinates, xp)

    @xp_function
    def get_inter_edges_weights(self, dim, metric='square', xp=np):
        inter_column_cost = move2gpu(self.inter_column_cost, xp)
        nodes_per_voxel = move2gpu(self.nodes_per_voxel, xp)
        intra_column_mask = move2gpu(self.intra_column_mask, xp)
        nodes_id = move2gpu(self.nodes_id, xp)
        coordinates = move2gpu(self.coordinates, xp)

        # Find adjacent coordinates
        helper_mask = (coordinates[:, dim] < self.voxel_num[dim]-1)
        coordinates_adjacent = coordinates[helper_mask]
        coordinates_adjacent[:, dim] += 1
        voxel_id_adjacent = self._ravel_index(coordinates_adjacent)
        voxel_id_rev = move2gpu(self.voxel_id_rev, xp)
        index_adjacent = voxel_id_rev[voxel_id_adjacent]
        helper_mask[helper_mask] = (index_adjacent != len(self.voxel_id))
        index_adjacent = index_adjacent[index_adjacent != len(self.voxel_id)]
        # Mask arrays with regard to the adjacent coordinate
        inter_column_cost_adjacent = inter_column_cost[index_adjacent, :]
        nodes_id_adjacent = nodes_id[index_adjacent, :]
        nodes_per_voxel_adjacent = nodes_per_voxel[index_adjacent]
        intra_column_mask_adjacent = intra_column_mask[index_adjacent, :]

        inter_column_cost = inter_column_cost[helper_mask]
        nodes_id = nodes_id[helper_mask]
        nodes_per_voxel = nodes_per_voxel[helper_mask]
        intra_column_mask = intra_column_mask[helper_mask]

        if self.voxel_weighting_inter_column:
            voxel_weighting = move2gpu(self.voxel_weighting, xp)
            voxel_weighting = xp.repeat(voxel_weighting[:, xp.newaxis],
                                        intra_column_mask.shape[-1], axis=-1)
            voxel_weighting = xp.minimum(voxel_weighting[index_adjacent],
                                         voxel_weighting[helper_mask])
        else:
            voxel_weighting = xp.ones_like(intra_column_mask)
        isotropic_scaling = self.isotropic_scaling[dim]

        if not self.chunk_mode:
            chunksize = nodes_per_voxel.shape[0]
        else:
            chunksize = self.chunksize
        self.chunk_mode = False
        reduce_gpu_mem = False
        # Calculate edges weights based on the fieldmap-frequency distance
        while True:
            try:
                edges = ChunkArr('uint32', reduce_gpu_mem, shape=(0, 4), xp=xp)
                tedges = ChunkArr('uint32', reduce_gpu_mem, shape=(0, 3), xp=xp)
                if chunksize == 0 or nodes_per_voxel.shape[0] == 0:
                    return edges.to_numpy(), tedges.to_numpy()

                num_chunks = int(np.ceil(nodes_per_voxel.shape[0] / chunksize))

                for i in range(num_chunks):
                    edges, tedges = \
                        self.__chunk_func_get_inter_edges_weights(chunksize, i,
                                                                  num_chunks, nodes_id,
                                                                  nodes_id_adjacent,
                                                                  inter_column_cost,
                                                                  inter_column_cost_adjacent,
                                                                  intra_column_mask,
                                                                  intra_column_mask_adjacent,
                                                                  nodes_per_voxel,
                                                                  nodes_per_voxel_adjacent,
                                                                  isotropic_scaling,
                                                                  edges, tedges, metric,
                                                                  voxel_weighting, xp)
                self.chunksize = chunksize
                return edges.to_numpy(), tedges.to_numpy()
            except CUDA_OutOfMemory:
                reduce_gpu_mem = True
                chunksize = chunksize // 2
                self.chunk_mode = True

    def __chunk_func_get_inter_edges_weights(self, chunksize, i, num_chunks,
                                             nodes_id, nodes_id_adjacent,
                                             inter_column_cost,
                                             inter_column_cost_adjacent,
                                             intra_column_mask,
                                             intra_column_mask_adjacent,
                                             nodes_per_voxel,
                                             nodes_per_voxel_adjacent,
                                             isotropic_scaling,
                                             edges, tedges, metric,
                                             voxel_weighting, xp):
        nodes_id_chunk = get_arr_chunks(chunksize, nodes_id, i, num_chunks)
        nodes_id_adjacent_chunk = get_arr_chunks(chunksize, nodes_id_adjacent,
                                                 i, num_chunks)
        inter_column_cost_chunk = get_arr_chunks(chunksize, inter_column_cost,
                                                 i, num_chunks)
        inter_column_cost_adjacent_chunk = get_arr_chunks(chunksize,
                                                          inter_column_cost_adjacent,
                                                          i, num_chunks)
        intra_column_mask_chunk = get_arr_chunks(chunksize, intra_column_mask,
                                                 i, num_chunks)
        intra_column_mask_adjacent_chunk = get_arr_chunks(chunksize,
                                                          intra_column_mask_adjacent,
                                                          i, num_chunks)
        nodes_per_voxel_chunk = get_arr_chunks(chunksize, nodes_per_voxel, i,
                                               num_chunks)
        nodes_per_voxel_adjacent_chunk = get_arr_chunks(chunksize,
                                                        nodes_per_voxel_adjacent,
                                                        i, num_chunks)
        voxel_weighting_chunk = get_arr_chunks(chunksize, voxel_weighting, i,
                                               num_chunks)

        a1 = inter_column_cost_chunk[intra_column_mask_chunk]
        helper_mask3 = intra_column_mask_chunk.copy()
        helper_mask3[:, 0] = False
        a2 = xp.where(helper_mask3[intra_column_mask_chunk],
                      xp.concatenate((xp.array([0]), a1[:-1])),
                      xp.zeros_like(a1))
        a3 = xp.repeat((inter_column_cost_adjacent_chunk[xp.arange(intra_column_mask_chunk.shape[0]),
                                                         nodes_per_voxel_adjacent_chunk-1])
                        [:, xp.newaxis], intra_column_mask_chunk.shape[1], axis=-1)[intra_column_mask_chunk]
        w = self._g2(a1, a2, a3, metric) * isotropic_scaling * \
            self.inter_column_scaling * voxel_weighting_chunk[intra_column_mask_chunk]
        weight = w[w > 0]
        weight[weight > INFTY] = INFTY
        tedges1 = xp.zeros((len(weight), 3), dtype=xp.uint32)
        tedges1[:, 0] = nodes_id_chunk[intra_column_mask_chunk][w > 0]
        tedges1[:, 2] = xp.around(weight)
        a1 = inter_column_cost_adjacent_chunk[intra_column_mask_adjacent_chunk]
        helper_mask2 = intra_column_mask_adjacent_chunk.copy()
        helper_mask2[:, 0] = False
        a2 = xp.where(helper_mask2[intra_column_mask_adjacent_chunk],
                      xp.concatenate((xp.array([0]), a1[:-1])),
                      xp.zeros_like(a1))
        a3 = xp.repeat((inter_column_cost_chunk[xp.arange(intra_column_mask_chunk.shape[0]),
                                                nodes_per_voxel_chunk-1])
                        [:, xp.newaxis], intra_column_mask_chunk.shape[1], axis=-1)[intra_column_mask_adjacent_chunk]
        w = self._g2(a1, a2, a3, metric) * isotropic_scaling * self.inter_column_scaling * \
            voxel_weighting_chunk[intra_column_mask_adjacent_chunk]
        weight = w[w > 0]
        weight[weight > INFTY] = INFTY
        tedges2 = xp.zeros((len(weight), 3), dtype=xp.uint32)
        tedges2[:, 0] = nodes_id_adjacent_chunk[intra_column_mask_adjacent_chunk][w > 0]
        tedges2[:, 2] = xp.around(weight)
        tedges1 = xp.concatenate((tedges1, tedges2))

        # Calculate edges between nodes
        edges1 = xp.zeros((0, 4), dtype=xp.uint32)
        # Use for-loop to reduce memory requirements
        for i in range(intra_column_mask_chunk.shape[1]):
            a1 = inter_column_cost_chunk[intra_column_mask_chunk]
            a2 = xp.where(helper_mask3[intra_column_mask_chunk],
                          xp.concatenate((xp.array([0]), a1[:-1])),
                          xp.zeros_like(a1))
            a3 = xp.repeat((inter_column_cost_adjacent_chunk[xp.arange(intra_column_mask_chunk.shape[0]), i])
                           [:, xp.newaxis], intra_column_mask_chunk.shape[1],
                           axis=-1)[intra_column_mask_chunk]
            if i > 0:
                a4 = xp.repeat((inter_column_cost_adjacent_chunk[xp.arange(intra_column_mask_chunk.shape[0]),
                                                                i-1])[:, xp.newaxis],
                                intra_column_mask_chunk.shape[1], axis=-1)[intra_column_mask_chunk]
            else:
                a4 = xp.zeros_like(a1)
            tmp_edges = xp.zeros((len(a1), 4), dtype=xp.uint32)
            if i > 0:
                w = xp.zeros_like(a1, dtype=xp.float32)
                w[a3 > 0] = (self._g(a1, a2, a3, a4, metric) * isotropic_scaling
                             * self.inter_column_scaling
                             * voxel_weighting_chunk[intra_column_mask_chunk])[a3 > 0]
                weight = w[w > 0]
                weight[weight > INFTY] = INFTY
                tmp_edges[w > 0, 0] = nodes_id_chunk[intra_column_mask_chunk][w > 0]
                tmp_edges[w > 0, 1] = xp.repeat((nodes_id_adjacent_chunk[xp.arange(intra_column_mask_chunk.shape[0]),
                                                                            i])[:, xp.newaxis],
                                                intra_column_mask_chunk.shape[1], axis=-1)[intra_column_mask_chunk][w > 0]
                tmp_edges[w > 0, 2] = xp.around(weight)
            w = xp.zeros_like(a1, dtype=xp.float32)
            w[(a2 > 0) & (a3 > 0)] = (self._g(a3, a4, a1, a2, metric) * isotropic_scaling \
                                      * self.inter_column_scaling
                                      * voxel_weighting_chunk[intra_column_mask_chunk])[(a2 > 0) & (a3 > 0)]
            weight = w[w > 0]
            weight[weight > INFTY] = INFTY
            tmp_edges[w > 0, 0] = nodes_id_chunk[intra_column_mask_chunk][w > 0]
            tmp_edges[w > 0, 1] = xp.repeat((nodes_id_adjacent_chunk[xp.arange(intra_column_mask_chunk.shape[0]),
                                                                        i])[:, xp.newaxis],
                                            intra_column_mask_chunk.shape[1], axis=-1)[intra_column_mask_chunk][w > 0]
            tmp_edges[w > 0, 3] = xp.around(weight)
            tmp_edges = tmp_edges[xp.nonzero(xp.sum(tmp_edges, axis=-1))]
            edges1 = xp.concatenate((edges1, tmp_edges))

        edges.concatenate(edges1)
        tedges.concatenate(tedges1)
        return edges, tedges

    def _ravel_index(self, coordinates):
        x, y, z = self.voxel_num
        return coordinates[:, 2] + z * coordinates[:, 1] + z * y * coordinates[:, 0]

    def _g2(self, a1, a2, a3, metric):
        func = getattr(self, f'_{metric}')
        return func(a1, a3) - func(a2, a3)

    def _g(self, a1, a2, a3, a4, metric):
        func = getattr(self, f'_{metric}')
        return func(a1, a4) - func(a2, a4) - func(a1, a3) + func(a2, a3)

    @xp_function
    def _square(self, r1, r2, xp=np):
        value = xp.zeros_like(r1, xp.float32)
        value[r1 > r2] = xp.square(xp.abs(r1-r2))[r1 > r2]
        return value

    @xp_function
    def _abs(self, r1, r2, xp=np):
        value = xp.zeros_like(r1, xp.float32)
        value[r1 > r2] = xp.abs(r1-r2)[r1 > r2]
        return value
