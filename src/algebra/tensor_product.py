"""Implementation of tensor product Lie algebras and coefficients for decompositions of tensor products of
irreducible representations"""

import logging
import numpy as np
from src.algebra import algebra_base, cartan, representations, weyl_group, utils

log = logging.getLogger('logger')

_tensor_char = chr(10754)


class TensorLieAlgebra(algebra_base.Algebra):

    def __init__(self, algebra_list):
        """Initialize the algebra of a tensor product by combining the properties of the list of algebras"""
        log.info('Initialize tensor product of algebras ' + str(algebra_list))
        self._rank = sum([algebra.rank for algebra in algebra_list])
        self._cartan_matrix = cartan.diagonal_join([algebra.cartan_matrix for algebra in algebra_list])
        self._quadratic_form_matrix = cartan.diagonal_join([algebra.quadratic_form_matrix for algebra in algebra_list])
        self._name_list = [algebra.name for algebra in algebra_list]
        self._weyl_order = np.prod([algebra.weyl_order for algebra in algebra_list])
        self._highest_root = np.concatenate([algebra.highest_root for algebra in algebra_list])
        self._algebra_dimension = np.prod([algebra.algebra_dimension for algebra in algebra_list])
        self._dual_coxeter_number = sum([algebra.dual_coxeter_number - 1 for algebra in algebra_list]) + 1
        self._dynkin_diagram = ""
        for algebra in algebra_list[:-1]:
            self._dynkin_diagram += algebra.dynkin_diagram + "\n"
        self._dynkin_diagram += algebra_list[-1].dynkin_diagram

    @property
    def name(self):
        """Return the algebra name """
        full_name = ''
        for name in self._name_list[:-1]:
            full_name += name + " " + _tensor_char + " "
        return full_name + self._name_list[-1]

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return self._weyl_order

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return self._highest_root

    @property
    def algebra_dimension(self):
        """Get the dimension of the algebra"""
        return self._compute_algebra_dimension()

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        return self._dual_coxeter_number

    @property
    def dynkin_diagram(self):
        """Plot the Dynkin diagram"""
        return self._dynkin_diagram


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
