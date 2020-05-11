"""This file defines and implements classes describing all of the semi-simple Lie-algebras"""

import numpy as np
import math
from . import algebra_base, cartan, roots, utils, tensor_product


def get_valid_algebras():
    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E, 'F': F, 'G': G}


def build(cartan_classes, ranks):
    """A factory to construct Lie algebra objects of the given Cartan class and rank"""
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
            print("Using that D2 [SO(4)] is isomorphic to A1 [SU(2)] " + tensor_product._tensor_char + " A1 [SU(2)]")
            new_cartan_classes += ['A', 'A']
            new_ranks += [1, 1]
        elif cartan_class == 'B' and rank == 1:
            print("Using that B1 [SO(3)] is isomorphic to A1 [SU(2)]")
            new_cartan_classes += ['A']
            new_ranks += [1]
        elif cartan_class == 'D' and rank == 3:
            print("Using that D3 [SO(6)] is isomorphic to A3 [SU(4)]")
            new_cartan_classes += ['A']
            new_ranks += [3]
        elif cartan_class == 'E' and rank == 3:
            print("Using that E3 is isomorphic to A2 [SU(3)] " + tensor_product._tensor_char + " A1 [SU(2)]")
            new_cartan_classes += ['A', 'A']
            new_ranks += [2, 1]
        elif cartan_class == 'E' and rank == 4:
            print("Using that E4 is isomorphic to A4 [SU(5)]")
            new_cartan_classes += ['A']
            new_ranks += [4]
        elif cartan_class == 'E' and rank == 5:
            print("Using that E5 is isomorphic to D5 [SO(10)]")
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


class SemisimpleLieAlgebra(algebra_base.Algebra):

    _rank_restriction = None
    _cartan_class = None

    def __init__(self, rank):
        """Initialize algebra data"""
        self.__check_rank(rank)
        super(SemisimpleLieAlgebra, self).__init__(rank)
        if self.weyl_order > self._weyl_order_limit:
            print("Warning: Weyl order exceeds set limit for calculations")

    def _get_group_dimension(self):
        """Map the algebra rank to the group dimension"""
        return self.rank

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

    def _compute_dual_coxeter_number(self):
        """Compute and return the dual Coxeter number"""
        return np.sum(roots.comarks(self.highest_root, self.cartan_matrix, self.quadratic_form_matrix)) + 1


class A(SemisimpleLieAlgebra):

    _group_name = "SU"
    _cartan_class = 'A'
    _rank_restriction = 1

    def __init__(self, rank):
        super(A, self).__init__(rank)

    def _get_group_dimension(self):
        """Map the algebra rank to the group dimension"""
        return self.rank + 1

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        return cartan.default_matrix(rank)

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

    @property
    def algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number of roots in the root lattice"""
        return self.rank ** 2 + 2 * self.rank


class B(SemisimpleLieAlgebra):

    _group_name = "SO"
    _cartan_class = 'B'
    _rank_restriction = 2

    def __init__(self, rank):
        super(B, self).__init__(rank)

    def _get_group_dimension(self):
        """Map the algebra rank to the group dimension"""
        return 2 * self.rank + 1

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_matrix(rank)
        A[-2, -1] = -2
        return A

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

    @property
    def algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number of roots in the root lattice"""
        return 2 * self.rank ** 2 + self.rank


class C(SemisimpleLieAlgebra):

    _group_name = "SP"
    _cartan_class = 'C'
    _rank_restriction = 2

    def __init__(self, rank):
        super(C, self).__init__(rank)

    def _get_group_dimension(self):
        """Map the algebra rank to the group dimension"""
        return 2 * self.rank

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_matrix(rank)
        A[-1, -2] = -2
        return A

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

    @property
    def algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number of roots in the root lattice"""
        return 2 * self.rank ** 2 + self.rank


class D(SemisimpleLieAlgebra):

    _group_name = "SO"
    _cartan_class = 'D'
    _rank_restriction = 4

    def __init__(self, rank):
        super(D, self).__init__(rank)

    def _get_group_dimension(self):
        """"Map the algebra rank to the group dimension"""
        return 2 * self.rank

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_matrix(rank)
        A[-1, -2] = 0
        A[-2, -1] = 0
        A[-1, -3] = -1
        A[-3, -1] = -1
        return A

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

    @property
    def algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number of roots in the root lattice"""
        return 2 * self.rank ** 2 - self.rank


class E(SemisimpleLieAlgebra):

    _group_name = 'E'
    _cartan_class = 'E'
    _rank_restriction = [6, 7, 8]
    __weyl_orders = {6: 72 * math.factorial(6), 7: 72 * math.factorial(8),
                     8: 192 * math.factorial(10)}
    __dual_coxeter_numbers = {6: 12, 7: 18, 8: 30}
    __highest_roots = {6: np.array([0, 0, 0, 0, 0, 1], dtype=utils.itype),
                       7: np.array([1, 0, 0, 0, 0, 0, 0], dtype=utils.itype),
                       8: np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=utils.itype)}
    __algebra_dimensions = {6: 78, 7: 133, 8: 248}

    def __init__(self, rank):
        super(E, self).__init__(rank)

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_matrix(rank)
        A[-1, -2] = 0
        A[-2, -1] = 0
        if rank == 7:
            A[-1, -5] = -1
            A[-5, -1] = -1
        else:
            A[-1, -4] = -1
            A[-4, -1] = -1
        return A

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

    @property
    def algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number of roots in the root lattice"""
        return self.__algebra_dimensions[self.rank]


class F(SemisimpleLieAlgebra):

    _group_name = 'F'
    _cartan_class = 'F'
    _rank_restriction = [4]
    __weyl_orders = {4: 1152}
    __dual_coxeter_numbers = {4: 9}
    __algebra_dimensions = {4: 52}

    def __init__(self, rank):
        super(F, self).__init__(rank)

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_matrix(rank)
        A[1, 2] = -2
        return A

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

    @property
    def algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number of roots in the root lattice"""
        return self.__algebra_dimensions[self.rank]


class G(SemisimpleLieAlgebra):

    _group_name = 'G'
    _cartan_class = 'G'
    _rank_restriction = [2]
    __weyl_orders = {2: 12}
    __dual_coxeter_numbers = {2: 4}
    __algebra_dimensions = {2: 14}

    def __init__(self, rank):
        super(G, self).__init__(rank)

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        A = cartan.default_matrix(rank)
        A[0, 1] = -3
        return A
    
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

    @property
    def algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number of roots in the root lattice"""
        return self.__algebra_dimensions[self.rank]
