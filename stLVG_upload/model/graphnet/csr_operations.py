"""
Function that operate on CSR matrices
"""
import copy
import numpy as np
from scipy.sparse import csr_matrix
from .time_utils import timer


@timer
def row_normalize(graph: csr_matrix,
                  copy: bool = False,
                  verbose: bool = True):
    """
    Normalize a compressed sparse row (CSR) matrix by row
    """
    if copy:
        graph = graph.copy()

    data = graph.data

    for start_ptr, end_ptr in zip(graph.indptr[:-1], graph.indptr[1:]):

        row_sum = data[start_ptr:end_ptr].sum()

        if row_sum != 0:
            data[start_ptr:end_ptr] /= row_sum

        if verbose:
            print(f"normalized sum from ptr {start_ptr} to {end_ptr} "
                  f"({end_ptr - start_ptr} entries)",
                  np.sum(graph.data[start_ptr:end_ptr]))

    return graph