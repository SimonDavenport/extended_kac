"""This file contains functions implementing properties of the roots of a Lie algebra"""

import numpy as np
from . import utils


def to_simple_root_basis(alpha, A):
    """Convert the basis to that of integer multiples Q of simple roots.
    Use that A A.T Q = A alpha. where A is the Cartan matrix."""
    return np.linalg.solve(np.dot(A, A.T), np.dot(A, alpha))


def from_simple_root_basis(coefficients, A):
    """Convert a vector of simple root coefficients to a root in the fundamental basis"""
    return np.sum([coefficient * row for coefficient, row in zip(coefficients, A)], axis=0)


def inner_product(alpha, F, beta):
    """Compute the inner product of weights alpha and beta,
    using the quadratic form matrix F"""
    return np.dot(alpha, np.dot(F, beta))


def norm(alpha, F):
    """Compute the norm of a root"""
    return inner_product(alpha, F, alpha)


def comarks(highest_root, A, F):
    """The ith comark is (1/2) * a_i * (alpha_i, alpha_i)"""
    simple_root_norms = np.array([norm(simple_root, F) for simple_root in A])
    return np.round(0.5 * np.multiply(simple_root_norms, marks(highest_root, A))).astype(utils.itype)


def marks(highest_root, A):
    """The marks, a_i, are the coefficients of the highest root in the simple root basis"""
    return utils.round(to_simple_root_basis(highest_root, A))


def coroot(alpha, F):
    """Compute the coroot for a given root alpha"""
    return 2 * alpha / norm(alpha, F)


def get_positive_roots(root_system, A):
    """Extract the positive roots from the full root system. Positive roots are those roots
    made from linear combinations of simple roots with positive coefficients. """
    return root_system[[np.any(utils.round(to_simple_root_basis(root, A)) > 0) for root in root_system]]


def find_highest_root(root_system):
    """The unique root where the sum of coefficients is maximal. Find by searching the full root system"""
    return root_system[np.argmax(root_system.sum(axis=1))]


def weyl_vector(rank):
    """The weyl vector is a vector containing all 1s in the fundamental basis"""
    return np.ones(rank, dtype=utils.itype)
