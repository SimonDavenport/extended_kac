"""Implementation of embeddings of semisimple Lie algebras"""

import numpy as np
from . import semisimple_lie_algebra, simple_lie_algebra, utils, cartan, roots

_direct_sum_char = chr(10753)


def _find_algebra(matrix):
    valid_algebras = semisimple_lie_algebra.get_valid_algebras()
    rank = len(matrix)
    for cartan_label in valid_algebras.keys():
        try:
            algebra = valid_algebras[cartan_label](rank)
        except utils.RankError:
            continue
        ## TODO need to check for outer-automorphisms of diagrams, or find some other approach
        if np.all(algebra.cartan_matrix == matrix):
            print(algebra.name)
            return algebra


def _cartan_to_algebra(A):
    """Examine a block-diagonal Cartan matrix and convert to a list of corresponding Lie algebra classes"""
    algebra_list = []
    block_size = 0
    for row in range(1, len(A)):
        if np.all(A[row, :row] == 0) and np.all(A[:row, row] == 0):
            block_size = row - block_size
            block = A[row - block_size:row, row - block_size:row]
            print(block)
            algebra_list.append(_find_algebra(block))
            if row == len(A) - 1:
                print(block)
                block = A[-2:-1, -2:-1]
                algebra_list.append(_find_algebra(block))
    if len(algebra_list) == 0:
        algebra_list.append(_find_algebra(A))
    print(algebra_list)
    combined_name = ""
    for algebra in algebra_list[:-1]:
        combined_name += algebra.name + " " + _direct_sum_char + " "
    combined_name += algebra_list[-1].name
    return combined_name, algebra_list


def regular_embeddings(algebra, as_algebras=False):
    """Compute all possible regular embeddings of a given algebra from all possible root exclusions from the extended,
    Dynkin diagram. Simple embeddings are given by removing any two nodes with mark=1, and adding a U(1) algebra.
    Semisimple embeddings are given by removing a single node with prime number mark.
    """
    extended_cartan = cartan.extend_matrix(-algebra.highest_root, algebra.cartan_matrix, algebra.quadratic_form_matrix)
    print(extended_cartan)
    marks = roots.marks(algebra.highest_root, algebra.cartan_matrix)
    embeddings = {}
    small_primes = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    print(marks)
    for root1, mark1 in enumerate(marks):
        if mark1 in small_primes:
            truncated_cartan = np.delete(extended_cartan, [root1], 0)
            block_cartan = np.delete(truncated_cartan, [root1], 1)
            print(block_cartan)
            combined_name, algebra_list = _cartan_to_algebra(block_cartan)
            print(combined_name)
            if combined_name not in embeddings.keys():
                embeddings[combined_name] = algebra_list
        for root2, mark2 in enumerate(marks):
            if mark1 == 1 and mark2 == 1 and root2 > root1:
                truncated_cartan = np.delete(extended_cartan, [root1, root2], 0)
                block_cartan = np.delete(truncated_cartan, [root1, root2], 1)
                print(block_cartan)
                combined_name, algebra_list = _cartan_to_algebra(block_cartan)
                combined_name += " " + _direct_sum_char + " U(1)"
                algebra_list.append(simple_lie_algebra.U(1))
                if combined_name not in embeddings.keys():
                    embeddings[combined_name] = algebra_list
    return embeddings


def compute_branching_coefficients(irrep, embedding_algebra):
    """Compute the branching coefficient for embedding an irrep of some algebra into irreps of an embedding algebra"""



    return [(branching_irrep, branching_coefficient) for branching_irrep, branching_coefficient in embedding]


def compute_projection_matrix(algebra1, algebra2):
    """Compute the projection matrix, relating every weight in algebra2 to a weight in algebra1"""


def compute_index(algebra1, algebra2):
    """Compute the embedding index; the ratio of the square length of the projection of the highest root of algebra2
    to the square of the length of the highest root of algebra1"""


def branching_branching_coefficients_to_index(branching_coefficients, algebra1, algebra2):
    """Compute the embedding index directly from the branching coefficients and Dynkin indexes"""


def compute_tensor_product_branching_coefficient(irrep1, irrep2, irrep3, irrep4):
    """Compute the branching coefficient for embedding irrep3 x irrep4 of algebra 2 in irrep1 x irrep2 of algebra 1"""


def compute_maximal_regular_subalgebra(algebra):
    """Determine the subalgebra that allows the given algebra to be maximally embedded (ie. such that no further
    embeddings are possible beyound the maximal regular subalgebra)"""


def compute_maximal_special_subalgebra(algebra):
    pass


