"""Implements functions to generate root and weight lattices"""

import numpy as np
from . import utils, roots, weyl_group


class FreudenthalRecurser:
    """Implementation of the Freudenthal multiplicity formula"""

    def __init__(self, highest_weight, positive_roots, F):
        self._positive_roots = positive_roots
        self._F = F
        self._multiplicity = {utils.serialize(highest_weight): 1}
        self._weyl_vector = roots.weyl_vector(len(highest_weight))
        self._fixed_norm = roots.norm(self._weyl_vector + highest_weight, self._F)

    @property
    def multiplicity(self):
        """Return the current state of the multiplicity container"""
        return self._multiplicity

    def update_multiplicity(self, str_new_weight):
        """Update the multiplicity container for a new level of the weight lattice"""
        new_weight = utils.unserialize(str_new_weight)
        numerator = 0
        for positive_root in self._positive_roots:
            shifted_weight = new_weight + positive_root
            shifted_weight_str = utils.serialize(shifted_weight)
            while shifted_weight_str in self._multiplicity.keys():
                numerator += roots.inner_product(shifted_weight, positive_root, self._F) * self._multiplicity[shifted_weight_str]
                shifted_weight += positive_root
                shifted_weight_str = utils.serialize(shifted_weight)
        denominator = self._fixed_norm - roots.norm(self._weyl_vector + new_weight, self._F)
        self._multiplicity[str_new_weight] = np.int64(np.round(2 * numerator / denominator))


def sort_lattice(lattice):
    """Sort the lattice by its level and return its keys in order"""
    return sorted(lattice, key=lattice.__getitem__)


def generate_lattice(highest_weight, A, F):
    """Generate the weight lattice from a given highest weight state in the basis
    of simple roots. Coincides with the root lattice if the highest weight is chosen
    to be the highest root, i.e. the highest weight state in the adjoint representation."""
    highest_weight = utils.itype(highest_weight)
    root_system = weyl_group.generate_root_system(A, F)
    positive_roots = roots.get_positive_roots(root_system, A)
    recurser = FreudenthalRecurser(highest_weight, positive_roots, F)
    highest_weight_str = utils.serialize(highest_weight)
    this_level = {highest_weight_str: 0}
    lattice = this_level.copy()
    while len(this_level) > 0:
        next_level = {}
        for str_weight in this_level:
            start_weight = utils.unserialize(str_weight)
            for simple_root, coefficient in zip(A, start_weight):
                for step in range(1, coefficient + 1):
                    new_weight = start_weight - np.multiply(step, simple_root)
                    str_new_weight = utils.serialize(new_weight)
                    next_level[str_new_weight] =  this_level[str_weight] + step
        for str_new_weight in sort_lattice(next_level):
            recurser.update_multiplicity(str_new_weight)
            lattice[str_new_weight] = next_level[str_new_weight]
        this_level = next_level.copy()
    return [(utils.unserialize(state_str), recurser.multiplicity[state_str]) for state_str in sort_lattice(lattice)]
