"""This file defines and implements classes describing certian of the simple Lie-algebras"""

from . import algebra_base


class SimpleLieAlgebra(algebra_base.Algebra):
    _group_name = None

    def __init__(self, rank):
        super(SimpleLieAlgebra, self).__init__(rank)

class U(SimpleLieAlgebra):
    _group_name = 'U'

    def __init__(self, compactification_radius):
        self._compactification_radius = compactification_radius
        super(SimpleLieAlgebra, self).__init__(0)

    def _get_group_dimension(self):
        """Map the algebra compactification_radius to the group dimension"""
        return self.compactification_radius

