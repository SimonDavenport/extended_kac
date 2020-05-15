"""This file contains the base class for the general definition of an algebra"""

import numpy as np
import logging
from src.algebra import cartan, utils, weyl_group, lattice, roots, dynkin

log = logging.getLogger('logger')

class Algebra:

    _weyl_order_limit = 1e8
    _group_name = None

    def __init__(self, rank):
        """Default constructor"""
        self._rank = rank
        self._cartan_matrix = self._build_cartan_matrix(rank)
        self._modified_cartan_matrix = self._build_modified_cartan_matrix(rank)
        self._quadratic_form_matrix = cartan.get_quadratic_form_matrix(self._cartan_matrix)
        self._simplified_root_space_formula = self._build_simplified_root_space_formula(rank)
        self._override_simplified_formula = False

    @property
    def name(self):
        """Return the algebra name """
        return self._group_name + "(" + str(self._get_group_dimension()) + ")"

    def _get_group_dimension(self):
        """Get the group dimension associated with the algebra"""
        return None

    @property
    def rank(self):
        """Get the rank of the algebra"""
        return self._rank

    @property
    def cartan_matrix(self):
        """Return the Cartan matrix"""
        return self._cartan_matrix

    def _build_cartan_matrix(self, rank):
        """Build the Cartan matrix"""
        return np.array([], dtype=utils.itype)

    @property
    def modified_cartan_matrix(self):
        """Return the modified cartan matrix, which is the cartan matrix written in 
        a basis for the simple roots that is optimal for computing the actions 
        of the Weyl group"""
        return self._modified_cartan_matrix

    def _build_modified_cartan_matrix(self, rank):
        """Build the modified Cartan matrix"""
        return np.array([], dtype=utils.itype)

    @property
    def quadratic_form_matrix(self):
        """Return the quadratic form matrix"""
        return self._quadratic_form_matrix

    def _build_simplified_root_space_formula(self, rank):
        """A simple formula to generate all the roots given the modified Cartan matrix"""
        return None

    @property 
    def override_simplified_formula(self):
        """Use to override the simple formula for determining the root space e.g. for testing"""
        return self._override_simplified_formula

    @override_simplified_formula.setter
    def override_simplified_formula(self, value):
        """Use to override the simple formula for determining the root space e.g. for testing"""
        self._override_simplified_formula = value

    @property
    def root_space_order(self):
        """Return the order of the root space"""
        return None

    @property
    def weyl_order(self):
        """Return the order of the associated Weyl group"""
        return None

    def _check_weyl_order(self):
        """Raise an error message if the Weyl group is going to be too large to
        make calculations tractable"""
        if self.weyl_order is None:
            raise utils.WeylGroupError("Weyl group is not defined or infinite")
        if self.weyl_order > self._weyl_order_limit:
            raise utils.WeylGroupError("Weyl group too large")

    @property
    def weyl_group(self):
        """Generate and return a table of Weyl group elements in a compact format"""
        self._check_weyl_order()
        return weyl_group.generate_elements(self.cartan_matrix, serizlized=False)

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return []

    @property
    def root_system(self):
        """Generate and return the root system"""
        if self._simplified_root_space_formula is not None and not self._override_simplified_formula:
            log.debug("Simplified root system formula is available")
            return self._simplified_root_space_formula(self._modified_cartan_matrix)
        else:
            log.debug("No simplified root system formula is available (or applying override); enumerating roots directly")
            self._check_weyl_order()
            return weyl_group.generate_root_system(self.cartan_matrix, 
                                                   self.quadratic_form_matrix)

    @property
    def root_lattice(self):
        """Generate the root lattice - the weight lattice in the adjoiint representation. 
        It correpseonds to the root system, with all the multiplicity labels being 1"""
        positive_roots = roots.get_positive_roots(self.root_system, self.cartan_matrix)
        return lattice.generate_lattice(self.highest_root, positive_roots, 
                                        self.cartan_matrix, self.quadratic_form_matrix)

    def _compute_algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number 
        of roots in the root lattice"""
        return sum([mul for root, mul in self.root_lattice])

    @property
    def algebra_dimension(self):
        """Get the dimension of the algebra"""
        if self.root_space_order is None:
            return None
        else:
            return self.rank + self.root_space_order

    @property
    def dual_coxeter_number(self):
        """Return the dual Coxeter number"""
        raise NotImplementedError("No implementation of dual coxeter number")

    @property
    def weyl_vector(self):
        return roots.weyl_vector(self.rank)

    @property
    def dynkin_diagram(self):
        """Plot the Dynkin diagram"""
        return dynkin.draw_diagram(self._cartan_matrix)
