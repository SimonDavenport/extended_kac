"""This file defines and implements classes describing all of the semi-simple Lie-algebras"""

import logging
import numpy as np
import math
from sympy.utilities.iterables import multiset_permutations
from src.algebra import kac_moody_algebra, cartan, roots, utils, tensor_product

log = logging.getLogger()

def get_valid_algebras():
    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G}


def build(cartan_classes, ranks):
    """A factory to construct Lie algebra objects of the given Cartan class(es) and rank(s)"""
    log.info('Build semisimple Lie algebra(s): class(es) ' + str(cartan_classes) + ' rank(s) ' + str(ranks))
    if not hasattr(cartan_classes, '__iter__'):
        cartan_classes = [cartan_classes]
    if not hasattr(ranks, '__iter__'):
        ranks = [ranks]
    valid_algebras = get_valid_algebras()
    new_cartan_classes = []
    new_ranks = []
    for cartan_class, rank in zip(cartan_classes, ranks):
        if cartan_class not in valid_algebras.keys():
            raise RuntimeError("Cartan class " + str(cartan_class) + " not valid")
        if cartan_class == 'D' and rank == 2:
            log.info("Using that D2 [SO(4)] is isomorphic to A1 [SU(2)] " + tensor_product._tensor_char + " A1 [SU(2)]")
            new_cartan_classes += ['A', 'A']
            new_ranks += [1, 1]
        elif cartan_class == 'B' and rank == 1:
            log.info("Using that B1 [SO(3)] is isomorphic to A1 [SU(2)]")
            new_cartan_classes += ['A']
            new_ranks += [1]
        elif cartan_class == 'D' and rank == 3:
            log.info("Using that D3 [SO(6)] is isomorphic to A3 [SU(4)]")
            new_cartan_classes += ['A']
            new_ranks += [3]
        elif cartan_class == 'E' and rank == 3:
            log.info("Using that E3 is isomorphic to A2 [SU(3)] " + tensor_product._tensor_char + " A1 [SU(2)]")
            new_cartan_classes += ['A', 'A']
            new_ranks += [2, 1]
        elif cartan_class == 'E' and rank == 4:
            log.info("Using that E4 is isomorphic to A4 [SU(5)]")
            new_cartan_classes += ['A']
            new_ranks += [4]
        elif cartan_class == 'E' and rank == 5:
            log.info("Using that E5 is isomorphic to D5 [SO(10)]")
            new_cartan_classes += ['D']
            new_ranks += [5]
        else:
            new_cartan_classes += [cartan_class]
            new_ranks += [rank]
    algebra_list = [valid_algebras[cartan_class](rank) for cartan_class, rank in zip(new_cartan_classes, new_ranks)]
    if len(algebra_list) == 1:
        return algebra_list[0]
    else:
        return tensor_product.TensorLieAlgebra(algebra_list)


class SemisimpleLieAlgebra(kac_moody_algebra.KacMoodyAlgebra):

    _matrix_class = None
    _cartan_class = None
    _rank_restriction = None

    def __init__(self, rank):
        log.debug('Init semisimple Lie algebra ' + str(self._cartan_class) + str(rank))
        self.__check_rank(rank)
        super(SemisimpleLieAlgebra, self).__init__(rank)
        if self.weyl_order > self._weyl_order_limit:
            log.warn("Weyl order exceeds set limit for direct enumeration")
        log.info('Dynkin Diagram: ' + self.dynkin_diagram)

    def __check_rank(self, rank):
        """Perform type and bound checks on the rank"""
        rank_error = utils.RankError("Invalid rank " + str(rank) + " for " + self._cartan_class)
        if type(rank) is not int:
            raise rank_error
        if type(self._rank_restriction) is list:
            if rank not in self._rank_restriction:
                raise rank_error
        if type(self._rank_restriction) is int:
            if rank < self._rank_restriction:
                raise rank_error

    @property
    def type(self):
        return kac_moody_algebra.AlgebraType.FINITE

    @property
    def dual_coxeter_number_from_comarks(self):
        """General definition of dual Coxeter number for semisimple Lie algebras"""
        return np.sum(roots.comarks(self.highest_root, self.cartan_matrix, self.quadratic_form_matrix)) + 1

    @property
    def algebra_dimension(self):
        """General definition of algebra dimension for semisimple Lie algebras"""
        return self.rank + self.root_space_order

    @property
    def group_name(self):
        """Group name is of the form matrix class (dimension)"""
        return self._matrix_class + "(" + str(self.group_dimension) + ")"

    @property
    def weyl_vector(self):
        """General definition of Weyl vector for semisimple Lie algebras"""
        return roots.weyl_vector(self.rank)


class A(SemisimpleLieAlgebra):

    _matrix_class = "SU"
    _cartan_class = 'A'
    _rank_restriction = 1

    def __init__(self, rank):
        super(A, self).__init__(rank)

    @property
    def name(self):
        return self.group_name

    @property
    def group_dimension(self):
        """Map the algebra rank to the group dimension"""
        return self.rank + 1

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        return cartan.default_cartan_matrix(rank)

    def _build_orthogonal_cartan_matrix(self, rank):
        """Build orthogonal Cartan matrix"""
        B = cartan.default_orthogonal_cartan_matrix(rank)
        return np.vstack((B, np.array((rank-1) * [0] + [-1], dtype=utils.itype)))

    def _build_simplified_root_space_formula(self, rank):
        """Build simplified root space formula as a function of 
        the orthogonal Cartan matrix B"""
        return lambda B: np.array([B[i] - B[j] for i in range(0, rank+1) for j in range(0, rank+1) if i!=j])

    @property
    def root_space_order(self):
        """Return the order of the root space"""
        return self.rank * (self.rank + 1)

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return math.factorial(self.rank + 1)

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        return self.rank + 1

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        if self.rank == 1:
            return np.array([2], dtype=utils.itype)
        else:
            return np.array([1] + [0] * (self.rank - 2) + [1], dtype=utils.itype)


class B(SemisimpleLieAlgebra):

    _matrix_class = "SO"
    _cartan_class = 'B'
    _rank_restriction = 2

    def __init__(self, rank):
        super(B, self).__init__(rank)

    @property
    def name(self):
        return self.group_name

    def _get_group_dimension(self):
        """Map the algebra rank to the group dimension"""
        return 2 * self.rank + 1

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_cartan_matrix(rank)
        A[-2, -1] = -2
        return A

    def _build_orthogonal_cartan_matrix(self, rank):
        """Build orthogonal Cartan matrix"""
        B = cartan.default_orthogonal_cartan_matrix(rank)
        B[-1, -1] = 2
        return B

    def _build_simplified_root_space_formula(self, rank):
        """Build simplified root space formula as a function of 
        the orthogonal Cartan matrix B"""
        return lambda B: np.array([sign1 * B[i] + sign2 * B[j]
                                   for sign1 in [-1, 1] for sign2 in [-1, 1]
                                   for i in range(0, rank) for j in range(0, rank) if i>j] + 
                                  [sign * row for sign in [-1, 1] for row in B])

    @property
    def root_space_order(self):
        """Return the order of the root space"""
        return 2 * self.rank ** 2

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return (2 ** self.rank) * math.factorial(self.rank)

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        return 2 * self.rank - 1

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return np.array([0, 1] + [0] * (self.rank - 2), dtype=utils.itype)


class C(SemisimpleLieAlgebra):

    _matrix_class = "SP"
    _cartan_class = 'C'
    _rank_restriction = 2

    def __init__(self, rank):
        super(C, self).__init__(rank)

    @property
    def name(self):
        return self.group_name

    def _get_group_dimension(self):
        """Map the algebra rank to the group dimension"""
        return 2 * self.rank

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_cartan_matrix(rank)
        A[-1, -2] = -2
        return A

    def _build_orthogonal_cartan_matrix(self, rank):
        """Build orthogonal Cartan matrix"""
        return cartan.default_orthogonal_cartan_matrix(rank)

    def _build_simplified_root_space_formula(self, rank):
        """Build simplified root space formula as a function of 
        the orthogonal Cartan matrix B"""
        return lambda B: np.array([sign1 * B[i] + sign2 * B[j] 
                                   for sign1 in [-1, 1] for sign2 in [-1, 1]
                                   for i in range(0, rank) for j in range(0, rank) if i>j] + 
                                  [sign * 2 * row for sign in [-1, 1] for row in B])

    @property
    def root_space_order(self):
        """Return the order of the root space"""
        return 2 * self.rank ** 2

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return (2 ** self.rank) * math.factorial(self.rank)

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        return self.rank + 1

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return np.array([2] + [0] * (self.rank - 1), dtype=utils.itype)


class D(SemisimpleLieAlgebra):

    _matrix_class = "SO"
    _cartan_class = 'D'
    _rank_restriction = 4

    def __init__(self, rank):
        super(D, self).__init__(rank)

    @property
    def name(self):
        return self.group_name

    def _get_group_dimension(self):
        """"Map the algebra rank to the group dimension"""
        return 2 * self.rank

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_cartan_matrix(rank)
        A[-1, -2] = 0
        A[-2, -1] = 0
        A[-1, -3] = -1
        A[-3, -1] = -1
        return A

    def _build_orthogonal_cartan_matrix(self, rank):
        """Build orthogonal Cartan matrix"""
        B = cartan.default_orthogonal_cartan_matrix(rank)
        B[-2 ,-1] = 1
        return B

    def _build_simplified_root_space_formula(self, rank):
        """Build simplified root space formula as a function of 
        the orthogonal Cartan matrix B"""
        return lambda B: np.array([sign1 * B[i] + sign2 * B[j]
                                   for sign1 in [-1, 1] for sign2 in [-1, 1]
                                   for i in range(0, rank) for j in range(0, rank) if i>j])

    @property
    def root_space_order(self):
        """Return the order of the root space"""
        return 2 * self.rank * (self.rank - 1)

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return (2 ** (self.rank - 1)) * math.factorial(self.rank)

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        return 2 * self.rank - 2

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return np.array([0, 1] + [0] * (self.rank - 2), dtype=utils.itype)


class E(SemisimpleLieAlgebra):

    _matrix_class = 'E'
    _cartan_class = 'E'
    _rank_restriction = [6, 7, 8]
    __weyl_orders = {6: 72 * math.factorial(6), 7: 72 * math.factorial(8),
                     8: 192 * math.factorial(10)}
    __dual_coxeter_numbers = {6: 12, 7: 18, 8: 30}
    __highest_roots = {6: np.array([0, 0, 0, 1, 0, 0], dtype=utils.itype),
                       7: np.array([0, 0, 0, 0, 0, 0, 1], dtype=utils.itype),
                       8: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=utils.itype)}
    __root_space_orders = {6: 72, 7: 126, 8: 240}

    def __init__(self, rank):
        super(E, self).__init__(rank)

    @property
    def name(self):
        return self._matrix_class + str(self.rank)

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix - note following uncommon (but simpler) root convention
        following Ref. Lie Algebras of Finite and Affine Type by Roger Carter, Ch. 6."""
        A = cartan.default_cartan_matrix(rank)
        A[-3, -2] = 0
        A[-2, -3] = 0
        A[-4, -2] = -1
        A[-2, -4] = -1
        return A

    def _build_orthogonal_cartan_matrix(self, rank):
        """Build orthogonal Cartan matrix"""
        B = np.vstack((np.zeros((8-rank, rank)), cartan.default_orthogonal_cartan_matrix(rank).astype(np.float64)))
        B[: , -1] = -0.5
        B[-1 , -2] = 0
        B[-3 , -2] = 1
        return B

    def _build_sign_restrictions(self, rank):
        """Impose the construction rule prod_{1, .., 8} epsilon_i = 1 with
        E6: epsilon_1 = epsilon_2 = epsilon_8
        E7: epsilon_1 = epsilon_8 """
        base_set = np.array([[1, 1, 1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1, -1, -1]] + 
                            [perm for perm in multiset_permutations([1, 1, 1, 1, 1, 1, -1, -1])] +
                            [perm for perm in multiset_permutations([1, 1, 1, 1, -1, -1, -1, -1])] +
                            [perm for perm in multiset_permutations([1, 1, -1, -1, -1, -1, -1, -1])])
        if rank == 6:
            return base_set[np.logical_and(base_set[:, 0] == base_set[:, 1], base_set[:, 1] == base_set[:, 7])]
        if rank == 7:
            return base_set[base_set[:, 0] == base_set[:, 7]]
        if rank == 8:
            return base_set

    def _build_simplified_root_space_formula(self, rank):
        """Build simplified root space formula as a function of 
        the orthogonal Cartan matrix B"""
        min_index = 8 - rank
        max_index = 8 - int(rank!=8)
        include_18_term = rank==7
        return lambda B: np.array([np.round(sign1 * B[i] + sign2 * B[j]).astype(utils.itype) 
                                   for sign1 in [-1, 1] for sign2 in [-1, 1]
                                   for i in range(min_index, max_index) for j in range(min_index, max_index) if i>j] +
                                  [np.round(sign * (B[0] + B[7])).astype(utils.itype) 
                                   for sign in [-1, 1]] * include_18_term + 
                                  [np.round(0.5 * row).astype(utils.itype) 
                                   for row in np.dot(self._build_sign_restrictions(rank), B)])

    @property
    def root_space_order(self):
        """Return the order of the root space"""
        return self.__root_space_orders[self.rank]

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return self.__weyl_orders[self.rank]

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        return self.__dual_coxeter_numbers[self.rank]

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return self.__highest_roots[self.rank]


class F(SemisimpleLieAlgebra):

    _matrix_class = 'F'
    _cartan_class = 'F'
    _rank_restriction = [4]
    __weyl_orders = {4: 1152}
    __dual_coxeter_numbers = {4: 9}
    __root_space_orders = {4: 48}

    def __init__(self, rank):
        super(F, self).__init__(rank)

    @property
    def name(self):
        return self._matrix_class + str(self.rank)

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_cartan_matrix(rank)
        A[1, 2] = -2
        return A

    def _build_orthogonal_cartan_matrix(self, rank):
        """Build orthogonal Cartan matrix """
        B = cartan.default_orthogonal_cartan_matrix(rank)
        B[: , -1] = -1
        B[-1 , -1] = 1
        B[-2 , -2] = 2
        B[-1 , -2] = 0
        return B

    def _build_sign_restrictions(self, rank):
        """Signs can be +/- for the sum of the 4 roots"""
        base_set = np.array([[1, 1, 1, 1], [-1, -1, -1, -1]] + 
                            [perm for perm in multiset_permutations([1, 1, 1, -1])] +
                            [perm for perm in multiset_permutations([1, 1, -1, -1])] +
                            [perm for perm in multiset_permutations([1, -1, -1, -1])])
        return base_set

    def _build_simplified_root_space_formula(self, rank):
        """Build simplified root space formula as a function of 
        the orthogonal Cartan matrix B"""
        return lambda B: np.array([sign * row for sign in [-1, 1] for row in B] + 
                                  [sign1 * B[i] + sign2 * B[j]
                                   for sign1 in [-1, 1] for sign2 in [-1, 1]
                                   for i in range(0, rank) for j in range(0, rank) if i>j] +
                                  [np.round(0.5 * row).astype(utils.itype) 
                                   for row in np.dot(self._build_sign_restrictions(rank), B)])

    @property
    def root_space_order(self):
        """Return the order of the root space"""
        return self.__root_space_orders[self.rank]

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return self.__weyl_orders[self.rank]

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        return self.__dual_coxeter_numbers[self.rank]

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return np.array([1] + [0] * (self.rank - 1), dtype=utils.itype)


class G(SemisimpleLieAlgebra):

    _matrix_class = 'G'
    _cartan_class = 'G'
    _rank_restriction = [2]
    __weyl_orders = {2: 12}
    __dual_coxeter_numbers = {2: 4}
    __root_space_orders = {2: 12}

    def __init__(self, rank):
        super(G, self).__init__(rank)

    @property
    def name(self):
        return self._matrix_class + str(self.rank)

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_cartan_matrix(rank)
        A[0, 1] = -3
        return A

    @property
    def root_space_order(self):
        """Return the order of the root space"""
        return self.__root_space_orders[self.rank]

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return self.__weyl_orders[self.rank]

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        return self.__dual_coxeter_numbers[self.rank]

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return np.array([1] + [0] * (self.rank - 1), dtype=utils.itype)
