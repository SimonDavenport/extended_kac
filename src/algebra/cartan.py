"""This file contains utility functions to work with properties
of the Cartan matrix"""

import logging
import numpy as np
from src.algebra import utils, roots

log = logging.getLogger('logger')

def _toeplitz_matrix(dim, values, diagonals):
    """Build a toeplitz matrix given a list of values
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
    log.debug("Computing matrix of root connecting lines")
    connecting_lines = np.multiply(A, A.T)
    np.fill_diagonal(connecting_lines, 0)
    return connecting_lines


def root_ratios(A):
    """The ratio of root i size to root j size is A_ji  / A_ij"""
    log.debug("Computing matrix of root ratios")
    epsilon = 0.001
    round_fac = 6
    ratios_matrix = np.round(np.divide(A.T, A + epsilon) * round_fac) / round_fac
    return np.cumprod(np.triu(ratios_matrix, 1).sum(axis=0)[1:])


def default_cartan_matrix(rank):
    """Build a default Cartan matrix"""
    default_values = [-1, 2, -1]
    default_diagonals = [-1, 0, 1]
    A = _toeplitz_matrix(rank, default_values, default_diagonals)
    return np.array(A, dtype=utils.itype)


def default_modified_cartan_matrix(rank):
    """Return the modified cartan matrix, which is the cartan matrix written in 
    a basis for the simple roots that is optimal for computing the actions 
    of the Weyl group. Ref. Lie Algebras of Finite and Affine Type by 
    Roger Carter, Ch. 8. The modified Cartan B has rows encoding beta
    and is given by the inverse of the change of basis from simple roots to betas"""
    default_values = [-1, 1]
    default_diagonals = [-1, 0]
    B = _toeplitz_matrix(rank, default_values, default_diagonals)
    return np.array(B, dtype=utils.itype)


def get_quadratic_form_matrix(A):
    """Compute and return the quadratic form matrix F given the Cartan matrix A"""
    log.debug("Computing quadratic form matrix by directly inverting the Cartan matrix")
    if len(A):
        ratios = np.concatenate(([1.0], root_ratios(A)))
        normalized_ratios = ratios / max(ratios)
        cartan_inverse = np.linalg.inv(A)
        return np.multiply(cartan_inverse, normalized_ratios)
    else:
        return np.array([])


def extend_matrix(extension_root, A, F):
    """Construct the Cartan matrix extended by the given root"""
    log.debug("Extending Cartan matrix by root " + str(extension_root))
    extension_row = np.array([[utils.round(roots.inner_product(extension_root, F, roots.coroot(simple_root, F)))
                               for simple_root in A]])
    part_extended_cartan = np.vstack((A, extension_row))
    extension_col = [utils.round(roots.inner_product(simple_root, F, roots.coroot(extension_root, F))) for simple_root in A]
    extension_col += [utils.round(roots.norm(extension_root, F))]
    return np.hstack((part_extended_cartan, np.array([extension_col]).T))


def diagonal_join(matrix_list):
    """Combine a list of cartan matrices diagonally"""
    log.debug("Combining Cartan matrices into block diagonal format")
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

