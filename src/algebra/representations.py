"""Implementation of a class to contain data on a particular irreducible representation"""

import numpy as np
from src.algebra import lattice, roots, utils, young_tableaux


def _is_dominant(weight, positive_roots, F, prec = 10):
    """Test if the given weight is 'dominant', defined as having non-negative inner product with positive roots"""
    return np.all(np.round([roots.inner_product(weight, F, positive_root) for positive_root in positive_roots], prec) >= 0)


def _is_integeral(weight, root_system, F):
    """Test if the given weight is 'integral', defined as having integer-valued normalized inner product with all roots"""
    return utils.is_integer_vector([roots.inner_product(weight, F, roots.coroot(alpha, F)) for alpha in root_system])


class Irrep:

    __strict_checking = True

    def __init__(self, algebra, highest_weight_state):
        self._algebra = algebra
        self._check_highest_weight_state(highest_weight_state)
        self._hws = np.array(highest_weight_state, dtype=utils.itype)

    @property
    def algebra(self):
        """Return the associated algebra"""
        return self._algebra

    @property
    def highest_weight_state(self):
        """Return the highest weight state"""
        return self._hws

    def _check_highest_weight_state(self, hws):
        """Check the format of the highest weight vector"""
        if len(hws) != self._algebra.rank:
            raise RuntimeError("Highest weight state has invalid dimension")
        if self.__strict_checking:
            root_system = self.algebra.root_system
            positive_roots = roots.get_positive_roots(root_system, self.algebra.cartan_matrix)
            F = self.algebra.quadratic_form_matrix
            if not _is_integeral(hws, root_system, F):
                raise RuntimeError("Highest weight state must by integral")
            if not _is_dominant(hws, positive_roots, F):
                raise RuntimeError("Highest weight must be dominant")
        else:
            if not utils.is_integer_vector(hws):
                raise RuntimeError("Highest weight state must have integer coefficients")

    @property
    def dimension(self):
        """Compute and return the dimension of the representation using the Weyl formula, or in special cases
        Young tableaux enumeration"""
        if hasattr(self._algebra, "_cartan_class"):
            if self._algebra._cartan_class == 'A':
                frame = young_tableaux.Frame(young_tableaux.highest_weight_to_partition(self._hws))
                return frame.count_total_semistandard_tableaux(self._algebra.rank + 1)
        else:
            fixed_weight = self._hws + self._algebra.weyl_vector
            positive_roots = roots.get_positive_roots(self._algebra.root_system, self._algebra.cartan_matrix)
            F = self._algebra.quadratic_form_matrix
            return int(np.round(np.prod([roots.inner_product(fixed_weight, alpha, F) /
                                roots.inner_product(self._algebra.weyl_vector, alpha, F) for alpha in positive_roots])))

    @property
    def dynkin_index(self):
        """Compute and return the Dynkin index of the representation"""
        F = self._algebra.quadratic_form_matrix
        numerator = self.dimension * roots.inner_product(self._hws, self._hws + 2 * self._algebra.weyl_vector, F)
        denominator = 2 * self._algebra.algebra_dimension
        return numerator / denominator

    @property
    def weight_lattice(self):
        """Generate and return the weight lattice for the representation"""
        self._algebra._check_weyl_order()
        return lattice.generate_lattice(self._hws, self._algebra.cartan_matrix, self._algebra.quadratic_form_matrix)

    def character(self, op_weight):
        """Compute and return the character of the representation"""
        F = self._algebra.quadratic_form_matrix
        return sum(mul * np.exp(roots.inner_product(weight, op_weight, F)) for weight, mul in self.weight_lattice)