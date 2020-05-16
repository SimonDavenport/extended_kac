"""Unit tests for the lie algebra class"""

import unittest
import ddt
import numpy as np
import sys
import logging
from src.algebra import semisimple_lie_algebra, representations

log = logging.getLogger()

@ddt.ddt
class TestSemisimpleLieAlgebras(unittest.TestCase):

    def setUp(self):
        """Hack to record the output of TestCase which does not flush the first line in the output"""
        print("", file=log.handlers[0].stream)
        return super().setUp()

    def simplified_test_schedule(function):
        function = ddt.data(('A', 3), ('A', 4), ('A', 5), ('A', 6), ('B', 3), ('B', 4),
                            ('C', 3), ('C', 4), ('D', 4),  ('F', 4), ('G', 2), ('E', 6))(function)
        function = ddt.unpack(function)
        return function

    def test_schedule(function):
        function = ddt.data(('A', 3), ('A', 4), ('A', 5), ('A', 6), ('B', 3), ('B', 4),
                            ('C', 3), ('C', 4), ('D', 4),  ('F', 4), ('G', 2), ('E', 6), 
                            ('E', 7), ('E', 8))(function)
        function = ddt.unpack(function)
        return function

    @simplified_test_schedule
    def test_simplified_root_space_formula(self, cartan_label, rank):
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        roots = set([root.tostring() for root in algebra.root_system])
        algebra.override_simplified_formula = True
        roots1 = set([root.tostring() for root in algebra.root_system])
        assert roots == roots1

    @test_schedule
    def test_root_space_order(self, cartan_label, rank):       
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        assert algebra.root_space_order == len(algebra.root_system)

    @test_schedule
    def test_contains_highest_root(self, cartan_label, rank):
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        assert algebra.highest_root in algebra.root_system

    @test_schedule
    def test_positive_negative_root_space_subdivision(self, cartan_label, rank):
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        positive_roots = algebra.positive_roots
        negative_roots = -positive_roots
        proposed_root_system = np.concatenate((positive_roots, negative_roots))
        assert np.all(np.sort(algebra.root_system, axis=0) == np.sort(proposed_root_system, axis=0))

    @test_schedule
    def tets_positive_negative_root_lattice_subdivision(self, cartan_label, rank):
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        positive_roots = np.stack([root for root, mul in algebra.positive_root_lattice])
        negative_roots = -positive_roots
        zero_root = np.zeros(len(positive_roots[0]), dtype=utils.itype)
        proposed_root_lattice = np.concatenate((positive_roots, negative_roots, [zero_root]))
        assert np.all(np.sort(algebra.root_lattice, axis=0) == np.sort(proposed_root_lattice, axis=0))

    @test_schedule
    def test_root_lattice_multiplicity(self, cartan_label, rank):
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        root_lattice_mul = np.stack([mul for _, mul in algebra.root_lattice])
        assert len(root_lattice_mul[root_lattice_mul>1]) == 1

    @test_schedule
    def test_dual_coxeter_number_computation(self, cartan_label, rank):
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        assert algebra.dual_coxeter_number == algebra.dual_coxeter_number_from_comarks

    @test_schedule
    def test_algebra_dimension_computation(self, cartan_label, rank):
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        assert algebra.algebra_dimension == algebra.algebra_dimension_from_root_lattice
    
    @test_schedule
    def test_freudenthal_de_vries_strange_formula(self, cartan_label, rank):
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        assert np.abs(algebra.weyl_vector_norm - algebra.dual_coxeter_number * algebra.algebra_dimension / 12.0) < 1e-10

    @ddt.data ((4, (0, 1, 2, 3)), (5, (0, 1, 2, 1, 0)), (6, (0, 0, 2, 2, 0, 0)))
    @ddt.unpack
    def test_representation_total_dimension_matches_semistandard_tableau_count_for_A_type(self, rank, highest_weight_state):
        algebra = semisimple_lie_algebra.A(rank)
        rep = representations.Irrep(algebra, highest_weight_state)
        yt_repr_dimension = rep.dimension
        lattice = rep.weight_lattice
        lattice_repr_dimension = sum([mul for root, mul in lattice])
        assert yt_repr_dimension == lattice_repr_dimension
