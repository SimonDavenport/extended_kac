"""This file contains the base class for the general definition of a Kac Moody Algebra"""

import numpy as np
import enum
import abc
import logging
from src.algebra import cartan, utils, weyl_group, lattice, roots, dynkin

log = logging.getLogger()

class AlgebraType(enum.IntEnum):
    """Specified in order of generatily"""
    FINITE = 0
    AFFINE = 1
    INDEFINITE = 2

class KacMoodyAlgebra(abc.ABC):

    _weyl_order_limit = 1e8

    def __init__(self, rank):
        self._rank = rank
        self._cartan_matrix = None
        self._quadratic_form_matrix = None
        self._orthogonal_cartan_matrix = None
        self._simplified_root_space_formula = None
        self.override_simplified_formula = False

    @property 
    @abc.abstractmethod
    def type(self):
        """Which of the three types of Kac Moody Algebra"""
        pass

    @type.setter
    def type(self, value):
        raise NotImplementedError("Cannot overwrite type")

    @property
    @abc.abstractmethod
    def name(self):
        """Return the name of the algabra"""
        pass

    @name.setter
    def name(self, value):
        raise NotImplementedError("Cannot overwrite name")

    @property
    def rank(self):
        """Get the rank of the algebra"""
        return self._rank

    @rank.setter
    def rank(self, value):
        raise NotImplementedError("Cannot overwrite rank")

    @rank.setter
    def rank(self, value):
        raise NotImplementedError("Cannot overwrite rank")

    @property
    def cartan_matrix(self):
        """Return the (generalized) Cartan matrix"""
        if self._cartan_matrix is None:
            self._cartan_matrix = self._build_cartan_matrix(self.rank)
        return self._cartan_matrix

    @cartan_matrix.setter
    def cartan_matrix(self, value):
        raise NotImplementedError("Cannot overwrite (generalized) Cartan matrix")

    @abc.abstractmethod
    def _build_cartan_matrix(self, rank):
        pass

    @property
    def quadratic_form_matrix(self):
        """Return the quadratic form matrix; compute it if not done already"""
        if self._quadratic_form_matrix is None:
            self._quadratic_form_matrix = self._build_quadratic_form_matrix()
        return self._quadratic_form_matrix

    @quadratic_form_matrix.setter
    def quadratic_form_matrix(self, value):
        raise NotImplementedError("Cannot overwrite quadratic form matrix")

    def _build_quadratic_form_matrix(self):
        """Default method is to directly invert the Cartan matrix"""
        return cartan.get_quadratic_form_matrix(self.cartan_matrix)

    @property
    def orthogonal_cartan_matrix(self):
        """Return the (generalized) Cartan matrix in an orthogonal basis, 
        which is optimal for computing the actions of the Weyl group
        It is computed from the inverse of the change of basis from simple roots to betas
        Ref. Lie Algebras of Finite and Affine Type by Roger Carter, Ch. 8. """
        if self._orthogonal_cartan_matrix is None:
            self._orthogonal_cartan_matrix = self._build_orthogonal_cartan_matrix(self.rank)
        return self._orthogonal_cartan_matrix

    @orthogonal_cartan_matrix.setter
    def orthogonal_cartan_matrix(self, value):
        raise NotImplementedError("Cannot overwrite orthogonal Cartan matrix")

    def _build_orthogonal_cartan_matrix(self, rank):
        """Default implementation if the method is not applicable"""
        return None

    @property
    def simplified_root_space_formula(self):
        """A simple formula to generate all the roots given the orthogonal basis Cartan matrix. 
         Ref. Lie Algebras of Finite and Affine Type by Roger Carter, Ch. 8."""
        if self._simplified_root_space_formula is None:
            self._simplified_root_space_formula = self._build_simplified_root_space_formula(self.rank)
        return self._simplified_root_space_formula

    @simplified_root_space_formula.setter
    def simplified_root_space_formula(self, value):
        raise NotImplementedError("Cannot overwrite simplified root space formula")

    def _build_simplified_root_space_formula(self, rank):
        """Default implementation if the method is not applicable"""
        return None

    @property
    @abc.abstractmethod
    def root_space_order(self):
        """Get the order of the root space (number of roots)"""
        pass

    @root_space_order.setter
    def root_space_order(self, value):
        raise NotImplementedError("Cannot overwrite root space order")

    @property
    def algebra_dimension(self):
        """Get the dimension of the algebra"""
        return self.algebra_dimension_from_root_lattice

    @algebra_dimension.setter
    def algebra_dimension(self, value):
        raise NotImplementedError("Cannot overwrite algebra dimension")

    @property
    def algebra_dimension_from_root_lattice(self):
        """Compute and return the dimension of the algebra, generated from 
        the number of roots in the root lattice times their multiplicity"""
        return sum([mul for root, mul in self.root_lattice])

    @algebra_dimension_from_root_lattice.setter
    def algebra_dimension_from_root_lattice(self, value):
        raise NotImplementedError("Cannot overwrite algebra dimension form root lattice")

    @property
    @abc.abstractmethod
    def dual_coxeter_number(self):
        """Get the dual Coxeter number value"""
        pass

    @dual_coxeter_number.setter
    def dual_coxeter_number(self, value):
        raise NotImplementedError("Cannot overwrite dual Coxeter number")

    @property
    def dual_coxeter_number_from_comarks(self):
        """Use formula to get the dual Coxeter number from comarks, if it exists"""
        pass

    @dual_coxeter_number_from_comarks.setter
    def dual_coxeter_number_from_comarks(self, value):
        raise NotImplementedError("Cannot overwrite dual Coxeter number from comarks")

    @property
    @abc.abstractmethod
    def weyl_order(self):
        """Get the order of the associated Weyl group"""
        pass

    @weyl_order.setter
    def weyl_order(self, value):
        raise NotImplementedError("Cannot overwrite Weyl order")

    def _check_weyl_order(self):
        """Raise an error message if the Weyl group is going to be too large to
        make explicit calculations tractable"""
        if self.weyl_order > self._weyl_order_limit:
            raise utils.WeylGroupError("Weyl group too large (limit is " + str(self._weyl_order_limit) + ")")

    @property
    def weyl_group(self):
        """Generate and return a table of Weyl group elements in a compact format"""
        self._check_weyl_order()
        return weyl_group.generate_elements(self.cartan_matrix, serizlized=False)

    @weyl_group.setter
    def weyl_group(self, value):
        raise NotImplementedError("Cannot overwrite Weyl group")

    @property
    def root_system(self):
        """Generate and return the root system"""
        log.debug("Compute root system")
        if (self.simplified_root_space_formula is not None and self.orthogonal_cartan_matrix is not None 
            and not self.override_simplified_formula):
            log.debug("Simplified root system formula is available")
            return self.simplified_root_space_formula(self.orthogonal_cartan_matrix)
        else:
            log.debug("No simplified root system formula is available (or applying override); enumerating roots directly")
            self._check_weyl_order()
            return weyl_group.generate_root_system(self.cartan_matrix, self.quadratic_form_matrix)

    @root_system.setter
    def root_system(self, value):
        raise NotImplementedError("Cannot overwrite root system")

    @property 
    def positive_roots(self):
        """Generate and return the positive roots"""
        return roots.get_positive_roots(self.root_system, self.cartan_matrix)
    
    @positive_roots.setter
    def positive_roots(self, value):
        raise NotImplementedError("Cannot overwrite positive roots")

    @property
    def highest_root(self):
        """It is the root theta = sum_i a_i alpha_i where if all other roots are sum_i k_i alpha_i,
        then k_i <= a_i. Ref. Lie Algebras of Finite and Affine Type by Roger Carter, page 251 """
        return roots.find_highest_root(self.root_system, self.cartan_matrix)

    @highest_root.setter
    def highest_root(self, value):
        raise NotImplementedError("Cannot overwrite highest root")

    @property
    def root_lattice(self):
        """Generate the root lattice i.e. the weight lattice in the adjoint representation. 
        A lattice is represented as a list of tuples containing weights and their multiplicities"""
        positive_roots = roots.get_positive_roots(self.root_system, self.cartan_matrix)
        return lattice.generate_lattice(self.highest_root, positive_roots, 
                                        self.cartan_matrix, self.quadratic_form_matrix)

    @root_lattice.setter
    def root_lattice(self, value):
        raise NotImplementedError("Cannot overwrite root lattice")

    @property 
    def positive_root_lattice(self):
        """Generate just the positive roots from the root lattice"""
        roots = np.stack([root for root, _ in self.root_lattice])
        multiplicities = [mul for _, mul in self.root_lattice]
        positive_roots = [root for root in roots.get_positive_roots(roots, algebra.cartan_matrix)]
        multiplicities_of_positive_roots = multiplicities[[np.any(root==positive_roots) for root in roots]]
        return [(root, mul) for root, mul in zip(positive_roots, multiplicities_of_positive_roots)]

    @positive_root_lattice.setter
    def positive_root_lattice(self, value):
        raise NotImplementedError("Cannot overwrite positive root lattice")

    @property
    @abc.abstractmethod
    def weyl_vector(self):
        pass

    @weyl_vector.setter
    def weyl_vector(self, value):
        raise NotImplementedError("Cannot overwrite Weyl vector")

    @property
    def weyl_vector_norm(self):
        """Norm of the Weyl vector appears in teh Freudenthal-de Vries strange formula"""
        return roots.norm(self.weyl_vector, self.quadratic_form_matrix)

    @weyl_vector_norm.setter
    def weyl_vector_norm(self, value):
        raise NotImplementedError("Cannot overwrite norm of Weyl vector")

    @property
    def dynkin_diagram(self):
        """Return UTF-8 formatted representaiton of the Dynkin diagram"""
        return dynkin.draw_diagram(self.cartan_matrix)

    @dynkin_diagram.setter
    def dynkin_diagram(self, value):
        raise NotImplementedError("Cannot overwrite Dynkin diagram")

    @property
    def group_name(self):
        """Return the associated group name e.g. SU(2) if there is one"""
        pass

    @group_name.setter
    def group_name(self, value):
        raise NotImplementedError("Cannot overwrite group name")

    @property
    def group_dimension(self):
        """Return the associated group dimension e.g. the 2 in SU(2) if there is one"""
        pass

    @group_dimension.setter
    def group_dimension(self, value):
        raise NotImplementedError("Cannot overwrite group dimension")
