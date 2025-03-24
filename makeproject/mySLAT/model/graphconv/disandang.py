from typing import Union, Tuple, List
import numpy as np

import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors

from .csr_operations import remove_greater_than, row_normalize, filter_by_rank_and_threshold


def generate_spatial_distance_graph(locations: np.ndarray,
                                    nbr_object: NearestNeighbors = None,
                                    num_neighbours: int = None,
                                    radius: Union[float, int] = None,
                                    ) -> csr_matrix:
    """
    generate a spatial graph with neighbours within a given radius
    """

    num_locations = locations.shape[0]

    if nbr_object is None:
        # set up neighbour object
        nbrs = NearestNeighbors(algorithm='ball_tree').fit(locations)
    else:  # use provided sklearn NN object
        nbrs = nbr_object

    if num_neighbours is None:
        # no limit to number of neighbours
        return nbrs.radius_neighbors_graph(radius=radius,
                                           mode="distance")

    else:
        assert isinstance(num_neighbours, int), (
            f"number of neighbours {num_neighbours} is not an integer"
        )

        graph_out = nbrs.kneighbors_graph(n_neighbors=num_neighbours,
                                          mode="distance")

        if radius is not None:
            assert isinstance(radius, (float, int)), (
                f"Radius {radius} is not an integer or float"
            )

            graph_out = remove_greater_than(graph_out, radius,
                                            copy=False, verbose=False)

        return graph_out
    
    #返回的是一个邻接矩阵，矩阵中的元素表示两个节点之间的距离




def theta_from_spatial_graph(locations: np.ndarray,
                             spatial_graph: csr_matrix,
                             ):
    """
    get azimuthal angles from spatial graph and coordinates
    (assumed dim 1: x, dim 2: y, dim 3: z...)

    returns CSR matrix with theta (azimuthal angles) as .data
    """

    theta_data = np.zeros_like(spatial_graph.data, dtype=np.float32)

    for n in range(spatial_graph.indptr.shape[0] - 1):
        ptr_start, ptr_end = spatial_graph.indptr[n], spatial_graph.indptr[n + 1]
        nbr_indices = spatial_graph.indices[ptr_start:ptr_end]

        self_coord = locations[[n], :]
        nbr_coord = locations[nbr_indices, :]
        relative_coord = nbr_coord - self_coord

        theta_data[ptr_start:ptr_end] = np.arctan2(
            relative_coord[:, 1], relative_coord[:, 0])

    theta_graph = spatial_graph.copy()
    theta_graph.data = theta_data

    return theta_graph

#得到角度矩阵