"""This file contains functions implementing properties of the roots of an affine Lie algebra"""

import roots


def affine_coroot(alpha_hat, F):
    """Compute the coroot for a given affine root alpha"""
    scale_factor = 2 / affine_inner_product(alpha_hat, F, alpha_hat)
    return (scale_factor * component for component in alpha_hat)


def affine_inner_product(alpha_hat, F, beta_hat):
    """Compute the inner product of affine weights alpha_hat and beta_hat,
    using the quadratic form matrix F"""
    alpha, k_alpha, n_alpha = alpha_hat
    beta, k_beta, n_beta = beta_hat
    return roots.inner_product(alpha, F, beta) + k_alpha * n_beta + k_beta * beta


def affine_norm(alpha_hat, F):
    """Compute the norm of an affine root"""
    return affine_inner_product(alpha_hat, F, alpha_hat)
