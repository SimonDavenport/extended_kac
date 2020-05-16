"""Implementation of tensor product Lie algebras and coefficients for decompositions of tensor products of
irreducible representations"""

import logging
import numpy as np
from src.algebra import kac_moody_algebra, cartan, representations, weyl_group, utils

log = logging.getLogger()

_tensor_char = chr(10754)


class TensorLieAlgebra(kac_moody_algebra.KacMoodyAlgebra):

    def __init__(self, algebra_list):
        """Initialize the algebra of a tensor product by combining the properties of the list of algebras"""
        log.info('Initialize tensor product of algebras ' + _tensor_char.join([algebra.name for algebra in algebra_list]))
        super(TensorLieAlgebra, self).__init__(sum([algebra.rank for algebra in algebra_list]))
        self._algebra_list = algebra_list

    def _build_cartan_matrix(self, rank):
        """Cartan matrix is a block diagoal of the Cartans of the underlying algebras"""
        return cartan.diagonal_join([algebra.cartan_matrix for algebra in self._algebra_list])

    def _build_quadratic_form_matrix(self):
        """Quadratic form matrix is a block diagoal of the quadratic forms of the underlying algebras"""
        return cartan.diagonal_join([algebra.quadratic_form_matrix for algebra in self._algebra_list])

    @property
    def type(self):
        """Select the most general type of the composing algebras"""
        return kac_moody_algebra.AlgebraType(np.max([algebra.type for algebra in self._algebra_list]))

    @property
    def name(self):
        """A combination of the names of the underlying algebras"""
        return _tensor_char.join([algebra.name for algebra in self._algebra_list])

    @property
    def root_space_order(self):
        """Given by the product of the root space orders of the composigng algebras"""
        return np.prod([algebra.root_space_order for algebra in self._algebra_list])

    @property
    def algebra_dimension(self):
        """Given by the product of the dimensions of the underlying algebras"""
        return np.prod([algebra.algebra_dimension for algebra in self._algebra_list])

    @property
    def dual_coxeter_number(self):
        """Compute the combined dual coexeter number"""
        return sum([algebra.dual_coxeter_number - 1 for algebra in self._algebra_list]) + 1

    @property
    def weyl_order(self):
        """Given by the product of the Weyl group orders of the composing algebras"""
        return np.prod([algebra.weyl_order for algebra in self._algebra_list])

    @property
    def highest_root(self):
        """Given by concatenating the highest roots of the composing algebras"""
        return np.concatenate([algebra.highest_root for algebra in self._algebra_list])

    @property
    def weyl_vector(self):
        """Given by concatenating the highest roots of the composing algebras"""
        return np.concatenate([algebra.weyl_vector for algebra in self._algebra_list])

    @property
    def dynkin_diagram(self):
        """Plot the Dynkin diagram"""
        return ("\n" + _tensor_char + "\n").join([algebra.dynkin_diagram for algebra in self._algebra_list])


def compute_irrep_decomposition(irrep1, irrep2, as_irreps=False):
    """Compute the tensor product decomposition coefficients for the product of the two given irreps"""
    algebra = irrep1.algebra
    coeffs = {}
    cache = {}
    wg = weyl_group.generate_elements(algebra.cartan_matrix, serialized=True)
    for weight, mul in irrep2.weight_lattice:
        shifted_weight = irrep1.highest_weight_state + algebra.weyl_vector + weight
        for weyl_element in wg:
            reflected_weight = weyl_group._element_to_reflection(weyl_element, shifted_weight, algebra.cartan_matrix,
                                                                 algebra.quadratic_form_matrix, cache) - algebra.weyl_vector
            if np.all(reflected_weight >= 0):
                term = mul * weyl_group.signature(weyl_element)
                reflected_weight_str = utils.serialize(reflected_weight)
                if reflected_weight_str not in coeffs:
                    coeffs[reflected_weight_str] = term
                else:
                    coeffs[reflected_weight_str] += term
                break
    result_as_array = [(utils.unserialize(hws_str), coeffs[hws_str]) for hws_str in coeffs.keys() if coeffs[hws_str] > 0]
    if as_irreps:
        result_as_irreps = [(representations.Irrep(algebra, hws).highest_weight_state, coeff) for hws, coeff in result_as_array]
        return result_as_irreps
    else:
        return result_as_array
