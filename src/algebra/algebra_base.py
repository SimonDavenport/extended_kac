"""This file contains the base class for the general definition of an algebra"""

import numpy as np
from . import cartan, utils, weyl_group, lattice, roots, dynkin


class Algebra:

    _weyl_order_limit = 100000
    _group_name = None

    def __init__(self, rank):
        """Default constructor"""
        self._rank = rank
        self._cartan_matrix = self._build_cartan_matrix(rank)
        self._quadratic_form_matrix = cartan.get_quadratic_form_matrix(self._cartan_matrix)

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
    def quadratic_form_matrix(self):
        """Return the quadratic form matrix"""
        return self._quadratic_form_matrix

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
        """Generate and return a table of Weyl group elements"""
        self._check_weyl_order()
        return weyl_group.generate_elements(self.cartan_matrix)

    @property
    def highest_root(self):
        """The unique root where the sum of coefficients is maximal"""
        return []

    @property
    def root_system(self):
        """Generate and return the root system"""
        self._check_weyl_order()
        return weyl_group.generate_root_system(self.cartan_matrix, self.quadratic_form_matrix)

    @property
    def root_lattice(self):
        """Generate the root lattice"""
        self._check_weyl_order()
        return lattice.generate_lattice(self.highest_root, self.cartan_matrix, self.quadratic_form_matrix)

    def _compute_algebra_dimension(self):
        """Compute and return the dimension of the algebra, i.e. the number of roots in the root lattice"""
        return sum([mul for root, mul in self.root_lattice])

    @property
    def algebra_dimension(self):
        """Get the dimension of the algebra"""
        return self._compute_algebra_dimension()

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
