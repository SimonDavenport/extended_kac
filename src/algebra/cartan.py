"""This file contains utility functions to work with properties
of the Cartan matrix"""

import numpy as np
from . import utils, roots


def _tridiagonal_matrix(dim, values, diagonals):
    """Build a tridagonal matrix given a list of values
    and diagonals; values are pasted along the diagonals"""
    matrix = np.diag(dim * [0])
    for value, diagonal in zip(values, diagonals):
        offset = abs(diagonal)
        if offset < dim:
            matrix += np.diag((dim-offset) * [value], diagonal)
    return matrix


def nbr_connecting_lines(A):
    """Get the number of lines connecting all roots.
    The # lines between i and j is A_ij * A_ji"""
    connecting_lines = np.multiply(A, A.T)
    np.fill_diagonal(connecting_lines, 0)
    return connecting_lines


def root_ratios(A):
    """The ratio of root i size to root j size is A_ji  / A_ij"""
    epsilon = 0.001
    round_fac = 6
    ratios_matrix = np.round(np.divide(A.T, A + epsilon) * round_fac) / round_fac
    return np.cumprod(np.triu(ratios_matrix, 1).sum(axis=0)[1:])


def default_matrix(rank):
    """Build a default Cartan matrix"""
    default_cartan_values = [-1, 2, -1]
    default_cartan_diagonals = [-1, 0, 1]
    A = _tridiagonal_matrix(rank, default_cartan_values, default_cartan_diagonals)
    return np.array(A, dtype=utils.itype)


def get_quadratic_form_matrix(A):
    """Compute and return the quadratic form matrix F given the Cartan matrix A"""
    if len(A):
        ratios = np.concatenate(([1.0], root_ratios(A)))
        normalized_ratios = ratios / max(ratios)
        cartan_inverse = np.linalg.inv(A)
        return np.multiply(cartan_inverse, normalized_ratios)
    else:
        return np.array([])


def extend_matrix(extension_root, A, F):
    """Construct the Cartan matrix extended by the given root"""
    extension_row = np.array([[utils.round(roots.inner_product(extension_root, F, roots.coroot(simple_root, F)))
                               for simple_root in A]])
    part_extended_cartan = np.vstack((A, extension_row))
    extension_col = [utils.round(roots.inner_product(simple_root, F, roots.coroot(extension_root, F))) for simple_root in A]
    extension_col += [utils.round(roots.norm(extension_root, F))]
    return np.hstack((part_extended_cartan, np.array([extension_col]).T))


def diagonal_join(matrix_list):
    """Combine a list of cartan matrices diagonally"""
    if len(matrix_list) == 1:
        return matrix_list[0]
    else:
        nrows = [block.shape[0] for block in matrix_list]
        ncols = [block.shape[1] for block in matrix_list]
        new_matrix = np.zeros((sum(nrows), sum(ncols)), dtype=matrix_list[0].dtype)
        rowstarts = np.cumsum(np.concatenate([[0], nrows]))
        colstarts = np.cumsum(np.concatenate([[0], ncols]))
        for rowstart, nrow, colstart, ncol, block in zip(rowstarts, nrows, colstarts, ncols, matrix_list):
            new_matrix[rowstart:rowstart + nrow, colstart:colstart + ncol] = block
        return new_matrix

