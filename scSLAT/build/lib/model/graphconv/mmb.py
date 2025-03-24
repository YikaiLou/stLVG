"""
Main BANKSY functions

Nigel 3 dec 2020

updated 4 mar 2022
updated 8 Sep 2022
"""
# import sys
# print(sys.path)
# sys.path.append(r'D:\ppppaper\SLAT\SLAT-main\SLAT-main\scSLAT\model\graphconv')

import mmb

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from typing import Union, Tuple, List

import scipy.sparse as sparse
from scipy.sparse import csr_matrix, issparse
from sklearn.neighbors import NearestNeighbors

import anndata

import igraph
# print(f"Using igraph version {igraph.__version__}")
import leidenalg
# print(f"Using leidenalg version {leidenalg.__version__}\n")

from .time_utils import timer
from .csr_operations import remove_greater_than, row_normalize, filter_by_rank_and_threshold


def gaussian_weight_1d(distance: float, sigma: float):
    """
    Calculate normalized gaussian value for a given distance from central point
    Normalized by root(2*pi) x sigma
    """
    return np.exp(-0.5 * distance ** 2 / sigma ** 2) / (sigma * np.sqrt(2 * np.pi))

def gaussian_weight_2d(distance: float, sigma: float):
    """
    Calculate normalized gaussian value for a given distance from central point
    Normalized by 2*pi*sigma-squared
    """
    sigma_squared = float(sigma) ** 2
    return np.exp(-0.5 * distance ** 2 / sigma_squared) / (sigma_squared * 2 * np.pi)

def plot_1d_gaussian(sigma: float, min_val: float, max_val: float, interval: float):
    """
    plot a 1d gaussian distribution, check if area sums to 1
    """
    x = np.arange(min_val, max_val, interval)
    y = gaussian_weight_1d(x, sigma)
    print(f"checking area : {np.sum(y * interval)} (should sum to 1)")
    plt.plot(x, y)

def p_equiv_radius(p: float, sigma: float):
    """
    find radius at which you eliminate fraction p
    of a radial gaussian probability distribution
    """
    assert p < 1.0, f"p was {p}, must be less than 1"
    return np.sqrt(-2 * sigma ** 2 * np.log(p))


#
# spatial graph generation
# ========================
#

@timer
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

@timer
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


@timer
def generate_spatial_weights_fixed_nbrs(locations: np.ndarray,
                                        m: int = 0,  # azimuthal transform order
                                        num_neighbours: int = 10,
                                        decay_type: str = "reciprocal",
                                        nbr_object: NearestNeighbors = None,
                                        verbose: bool = True,
                                        max_radius: int = None,
                                        ) -> Tuple[csr_matrix, csr_matrix, Union[csr_matrix, None]]:
    """
    generate a graph (csr format) where edge weights decay with distance
    """

    distance_graph = generate_spatial_distance_graph(
        locations,
        nbr_object=nbr_object,
        num_neighbours=num_neighbours * (m + 1),
        radius=max_radius,
    )

    if m > 0:
        theta_graph = theta_from_spatial_graph(locations, distance_graph)
    else:
        theta_graph = None

    graph_out = distance_graph.copy()

    # compute weights from nbr distances (r)
    # --------------------------------------

    if decay_type == "uniform":

        graph_out.data = np.ones_like(graph_out.data)

    elif decay_type == "reciprocal":

        graph_out.data = 1 / graph_out.data

    elif decay_type == "reciprocal_squared":

        graph_out.data = 1 / (graph_out.data ** 2)

    elif decay_type == "scaled_gaussian":

        indptr, data = graph_out.indptr, graph_out.data

        for n in range(len(indptr) - 1):

            start_ptr, end_ptr = indptr[n], indptr[n + 1]
            if end_ptr >= start_ptr:
                # row entries correspond to a cell's neighbours
                nbrs = data[start_ptr:end_ptr]
                median_r = np.median(nbrs)
                #### Changed here
                weights = np.exp(-(nbrs / median_r) ** 2)
                data[start_ptr:end_ptr] = weights

    elif decay_type == "ranked":

        linear_weights = np.exp(
            -1 * (np.arange(1, num_neighbours + 1) * 1.5 / num_neighbours) ** 2
        )

        indptr, data = graph_out.indptr, graph_out.data

        for n in range(len(indptr) - 1):

            start_ptr, end_ptr = indptr[n], indptr[n + 1]

            if end_ptr >= start_ptr:
                # row entries correspond to a cell's neighbours
                nbrs = data[start_ptr:end_ptr]

                # assign the weights in ranked order
                weights = np.empty_like(linear_weights)
                weights[np.argsort(nbrs)] = linear_weights

                data[start_ptr:end_ptr] = weights

    else:
        raise ValueError(
            f"Weights decay type <{decay_type}> not recognised.\n"
            f"Should be 'uniform', 'reciprocal', 'reciprocal_squared', "
            f"'scaled_gaussian' or 'ranked'."
        )

    # make nbr weights sum to 1
    graph_out = row_normalize(graph_out, verbose=verbose)

    # multiply by Azimuthal Fourier Transform
    if m > 0:
        graph_out.data = graph_out.data * np.exp(1j * m * theta_graph.data)

    return graph_out, distance_graph, theta_graph


@timer
def generate_spatial_weights_fixed_radius(locations: np.ndarray,
                                          p: float = 0.05,
                                          sigma: float = 100,
                                          decay_type: str = "gaussian",
                                          nbr_object: NearestNeighbors = None,
                                          max_num_neighbours: int = None,
                                          verbose: bool = True,
                                          ) -> Tuple[csr_matrix, csr_matrix]:
    """
    generate a graph (csr format) where edge weights decay with distance
    """

    if decay_type == "gaussian":
        r = p_equiv_radius(p, sigma)
        if verbose:
            print(f"Equivalent radius for removing {p} of "
                  f"gaussian distribution with sigma {sigma} is: {r}\n")
    else:
        raise ValueError(
            f"decay_type {decay_type} incorrect or not implemented")

    distance_graph = generate_spatial_distance_graph(locations,
                                                     nbr_object=nbr_object,
                                                     num_neighbours=max_num_neighbours,
                                                     radius=r)

    graph_out = distance_graph.copy()

    # convert distances to weights
    if decay_type == "gaussian":
        graph_out.data = gaussian_weight_2d(graph_out.data, sigma)
    else:
        raise ValueError(
            f"decay_type {decay_type} incorrect or not implemented")

    return row_normalize(graph_out, verbose=verbose), distance_graph


#
# Combining self / neighbours
# ===========================
#
def zscore(matrix: Union[np.ndarray, csr_matrix],
           axis: int = 0,
           ) -> np.ndarray:
    """
    Z-score data matrix along desired dimension
    """

    E_x = matrix.mean(axis=axis)

    if issparse(matrix):

        squared = matrix.copy()
        squared.data **= 2
        E_x2 = squared.mean(axis=axis)

    else:

        E_x2 = np.square(matrix).mean(axis=axis)

    variance = E_x2 - np.square(E_x)

    zscored_matrix = (matrix - E_x) / np.sqrt(variance)

    if isinstance(zscored_matrix, np.matrix):
        zscored_matrix = np.array(zscored_matrix)

    # Ensure that there are no NaNs
    # (which occur when all values are 0, hence variance is 0)
    zscored_matrix = np.nan_to_num(zscored_matrix)

    return zscored_matrix

def concatenate_all(matrix_list: List[Union[np.ndarray, csr_matrix]],
                    neighbourhood_contribution: float,
                    adata: anndata.AnnData = None,
                    ) -> np.ndarray:
    """
    Concatenate self- with neighbour- feature matrices
    converts all matrices are dense
    z-scores each matrix and applies relevant neighbour contributions (lambda)
    each higher k is given half the weighting of the previous one.
    """

    num_k = len(matrix_list) - 1

    scale_factors_squared = np.zeros(len(matrix_list))

    scale_factors_squared[0] = 1 - neighbourhood_contribution

    denom = 0
    for k in range(num_k):
        denom += 1 / (2 ** (k + 1))

    for k in range(num_k):
        scale_factors_squared[k + 1] = 1 / (2 ** (k + 1)) / denom * neighbourhood_contribution

    scale_factors = np.sqrt(scale_factors_squared)

    print(f"Scale factors squared: {scale_factors_squared}\nScale factors: {scale_factors}")

    scaled_list = []
    for n in range(len(matrix_list)):
        mat = matrix_list[n]
        if issparse(mat):
            mat = mat.todense()
        if np.iscomplexobj(mat):
            mat = np.absolute(mat)
        scaled_list.append(scale_factors[n] * zscore(mat, axis=0))

    concatenated_matrix = np.concatenate(scaled_list, axis=1)

    if isinstance(adata, anndata.AnnData):

        var_original = adata.var.copy()
        var_original["is_nbr"] = False
        var_original["k"] = -1  # k not applicable

        var_list = [var_original, ]
        for k in range(num_k):
            var_nbrs = adata.var.copy()
            var_nbrs["is_nbr"] = True
            var_nbrs["k"] = k
            var_nbrs.index += f"_nbr_{k}"
            var_list.append(var_nbrs)

        var_combined = pd.concat(var_list)

        return anndata.AnnData(concatenated_matrix, obs=adata.obs, var=var_combined)

    elif adata is None:
        return concatenated_matrix

    else:
        print("Adata type not recognised. Should be AnnData or None.")


@timer
def weighted_concatenate(cell_genes: Union[np.ndarray, csr_matrix],
                         neighbours: Union[np.ndarray, csr_matrix],
                         neighbourhood_contribution: float,
                         ) -> Union[np.ndarray, csr_matrix]:
    """
    Concatenate self- with neighbour- feature matrices
    with a given contribution towards disimilarity from the neighbour features (lambda).
    Assumes that both matrices have already been z-scored.
    Will do sparse concatenation if BOTH matrices are sparse.
    """
    cell_genes *= np.sqrt(1 - neighbourhood_contribution)
    neighbours *= np.sqrt(neighbourhood_contribution)

    if issparse(cell_genes) and issparse(neighbours):

        return sparse.hstack((cell_genes, neighbours))

    else:  # at least one is a dense array

        if issparse(cell_genes):
            cell_genes = cell_genes.todense()
        elif issparse(neighbours):
            neighbours = neighbours.todense()

        return np.concatenate((cell_genes, neighbours), axis=1)

def banksy_matrix_to_adata(banksy_matrix,
                           adata: anndata.AnnData,  # original adata object
                           ) -> anndata.AnnData:
    """
    convert a banksy matrix to adata object, by 
     - duplicating the original var (per-gene) annotations and adding "_nbr"
     - keeping the obs (per-cell) annotations the same as original anndata that banksy matrix was computed from
    """

    var_nbrs = adata.var.copy()
    var_nbrs.index += "_nbr"
    nbr_bool = np.zeros((var_nbrs.shape[0] * 2,), dtype=bool)
    nbr_bool[var_nbrs.shape[0]:] = True
    print("num_nbrs:", sum(nbr_bool))

    var_combined = pd.concat([adata.var, var_nbrs])
    var_combined["is_nbr"] = nbr_bool

    return anndata.AnnData(banksy_matrix, obs=adata.obs, var=var_combined)



@timer
def median_dist_to_nearest_neighbour(adata: anndata.AnnData,
                                     key: str = "coord_xy"):
    
    '''Finds and returns median cell distance in a graph'''
    nbrs = NearestNeighbors(algorithm='ball_tree').fit(adata.obsm[key])
    distances, indices = nbrs.kneighbors(n_neighbors=1)
    median_cell_distance = np.median(distances)
    print(f"\nMedian distance to closest cell = {median_cell_distance}\n")
    return nbrs