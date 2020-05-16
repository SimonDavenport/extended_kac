"""This file defines and implements classes describing twisted and
non-twisted affine Lie Algebras with both integer and fractional levels"""

import numpy as np
from src.algebra import kac_moody_algebra, cartan
import fractions


class AffineLieAlgebra(kac_moody_algebra.KacMoodyAlgebra):

    def __init__(self, semisimple_lie_algebra, level, is_twisted):
        """Perform some type checks and initialize algebra data"""
        if type(level) is int:
            pass
        elif type(level) is fractions.Fraction:
            raise NotImplementedError("Fractional level not implemented")
        else:
            raise RuntimeError("Invalid level " + str(level))
        if type(is_twisted) is not bool:
            raise RuntimeError("Invalid is_twisted " + str(is_twisted))
        self.__level = level
        self.__is_twisted = is_twisted
        self.__semisimple_lie_algebra = semisimple_lie_algebra
        self._cartan_matrix = self._build_extended_cartan_matrix()

    @property
    def name(self):
        """Return the Lie algebra name """
        return self.algebra_name + "(" + str(self.group_dimension) + ") level " + str(self.level)

    @property
    def group_name(self):
        """Return the name of the associated group"""
        return self.__lie_algebra.group_name

    @property
    def group_dimension(self):
        """Map the algebra rank to the group dimension"""
        return self.__lie_algebra.group_dimension

    @property
    def rank(self):
        return self.__lie_algebra.rank

    @property
    def level(self):
        return self.__level

    @property
    def is_twisted(self):
        return self.__is_twisted

    def _build_extended_cartan_matrix(self):
        """Build the Cartan matrix"""
        return cartan.extend_matrix(-self.highest_root, self.__lie_algebra.cartan_matrix,
                                    self.__lie_algebra.quadratic_form_matrix)

    @property
    def quadratic_form_matrix(self):
        """Compute and return the quadratic form matrix"""
        return self.__lie_algebra.quadratic_form_matrix

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        ## TODO define the extension comark
        extension_comark = None
        return self.__lie_algebra.dual_coxeter_number + extension_comark - 1


class A(AffineLieAlgebra):

    def __init__(self, rank, level, is_twisted=False):
        super(A, self).__init__(lie_algebra.A(rank), level, is_twisted)


class B(AffineLieAlgebra):

    def __init__(self, rank, level, is_twisted=False):
        super(B, self).__init__(lie_algebra.B(rank), level, is_twisted)


class C(AffineLieAlgebra):

    def __init__(self, rank, level, is_twisted=False):
        super(C, self).__init__(lie_algebra.C(rank), level, is_twisted)


class D(AffineLieAlgebra):

    def __init__(self, rank, level, is_twisted=False):
        super(D, self).__init__(lie_algebra.D(rank), level, is_twisted)


class E(AffineLieAlgebra):

    def __init__(self, rank, level, is_twisted=False):
        super(E, self).__init__(lie_algebra.E(rank), level, is_twisted)


class F(AffineLieAlgebra):

    def __init__(self, rank, level, is_twisted=False):
        super(F, self).__init__(lie_algebra.F(rank), level, is_twisted)


class G(AffineLieAlgebra):

    def __init__(self, rank, level, is_twisted=False):
        super(G, self).__init__(lie_algebra.G(rank), level, is_twisted)
