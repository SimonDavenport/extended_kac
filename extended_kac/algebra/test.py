"""Unit tests for the lie algebra class"""

import numpy as np
from . import semisimple_lie_algebra, roots, utils, representations


def _make_message(message, status):
    if status:
        print(message + "passed!")
    else:
        print(message + "failed!")
        raise RuntimeError("Unit test failure")


def run_unit_test(message_detail=False):
    """Run all tests and print whether they are passed or not"""
    passed_status = True
    test_schedule = (('A', 3), ('A', 4), ('A', 5), ('A', 6), ('B', 3), ('B', 4),
                     ('C', 3), ('C', 4), ('D', 4), ('E', 6), ('F', 4), ('G', 2))
    passed_status &= _contains_highest_root(test_schedule, message_detail)
    passed_status &= _positive_negative_root_space_subdivision(test_schedule, message_detail)
    passed_status &= _positive_negative_root_lattice_subdivision(test_schedule, message_detail)
    passed_status &= _root_lattice_multiplicity(test_schedule, message_detail)
    passed_status &= _dual_coxeter_number_computation(test_schedule, message_detail)
    passed_status &= _algebra_dimension_computation(test_schedule, message_detail)
    passed_status &= _strange_formula(test_schedule, message_detail)
    passed_status &= _representation_total_dimension(message_detail)
    _make_message("All tests: ", passed_status)
    return passed_status


def _contains_highest_root(test_schedule, message_detail):
    """Check root system contains expected highest root"""
    passed_status = True
    for cartan_label, rank in test_schedule:
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        passed_status &= algebra.highest_root in algebra.root_system
        if message_detail:
            _make_message(algebra.name + " ", passed_status)
        if not passed_status:
            break
    _make_message("Root space contains highest root test: ", passed_status)
    return passed_status


def _positive_negative_root_space_subdivision(test_schedule, message_detail):
    """Check that the root space subdivides symmetrically into positive and negative roots"""
    passed_status = True
    for cartan_label, rank in test_schedule:
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        root_system = algebra.root_system
        positive_roots = roots.get_positive_roots(root_system, algebra.cartan_matrix)
        negative_roots = -positive_roots
        proposed_root_system = np.concatenate((positive_roots, negative_roots))
        passed_status &= np.all(np.sort(root_system, axis=0) == np.sort(proposed_root_system, axis=0))
        if message_detail:
            _make_message(algebra.name + " ", passed_status)
        if not passed_status:
            break
    _make_message("Positive/negative root space subdivision test: ", passed_status)
    return passed_status


def _positive_negative_root_lattice_subdivision(test_schedule, message_detail):
    """Check that the root lattice subdivides symmetrically into positive and negative roots"""
    passed_status = True
    for cartan_label, rank in test_schedule:
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        root_lattice = np.stack([root for root, mul in algebra.root_lattice])
        positive_roots = roots.get_positive_roots(root_lattice, algebra.cartan_matrix)
        negative_roots = -positive_roots
        zero_root = np.zeros(len(positive_roots[0]), dtype=utils.itype)
        proposed_root_lattice = np.concatenate((positive_roots, negative_roots, [zero_root]))
        passed_status &= np.all(np.sort(root_lattice, axis=0) == np.sort(proposed_root_lattice, axis=0))
        if message_detail:
            _make_message(algebra.name + " ", passed_status)
        if not passed_status:
            break
    _make_message("Positive/negative root lattice subdivision test: ", passed_status)
    return passed_status


def _root_lattice_multiplicity(test_schedule, message_detail):
    """Check that the non-zero state multiplicity of the root lattice is always 1"""
    passed_status = True
    for cartan_label, rank in test_schedule:
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        root_lattice_mul = np.stack([mul for root, mul in algebra.root_lattice])
        passed_status &= len(root_lattice_mul[root_lattice_mul>1]) == 1
        if message_detail:
            _make_message(algebra.name + " ", passed_status)
        if not passed_status:
            break
    _make_message("Root lattice multiplicity test: ", passed_status)
    return passed_status


def _dual_coxeter_number_computation(test_schedule, message_detail):
    """Check that the computed value of the dual Coxeter number matches the tabulated value"""
    passed_status = True
    for cartan_label, rank in test_schedule:
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        passed_status &= algebra.dual_coxeter_number == algebra._compute_dual_coxeter_number()
        if message_detail:
            _make_message(algebra.name + " ", passed_status)
        if not passed_status:
            break
    _make_message("Dual Coxeter number test: ", passed_status)
    return passed_status


def _algebra_dimension_computation(test_schedule, message_detail):
    """Check that the computed value of the algebra dimension matches the tabulated value"""
    passed_status = True
    for cartan_label, rank in test_schedule:
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        passed_status &= algebra.algebra_dimension == algebra._compute_algebra_dimension()
        if message_detail:
            _make_message(algebra.name + " ", passed_status)
        if not passed_status:
            break
    _make_message("Algebra dimension test: ", passed_status)
    return passed_status


def _strange_formula(test_schedule, message_detail):
    """Confirm that the Freudenthal-de Vries strange formula is satisfied"""
    passed_status = True
    for cartan_label, rank in test_schedule:
        algebra = semisimple_lie_algebra.build(cartan_label, rank)
        lhs = roots.norm(roots.weyl_vector(rank)  , algebra.quadratic_form_matrix)
        rhs = algebra.dual_coxeter_number * algebra.algebra_dimension / 12.0
        passed_status &= np.abs(lhs - rhs) < 1e-10
        if message_detail:
            _make_message(algebra.name + " ", passed_status)
        if not passed_status:
            break
    _make_message("Freudenthal-de Vries strange formula test: ", passed_status)
    return passed_status


def _representation_total_dimension(message_detail):
    """Check that the total dimensions of the weight lattice matches the counting
    of semistandard Young tableaux (for A-type Lie algebras only)"""
    highest_weights = {4: [0, 1, 2, 3], 5: [0, 1, 2, 1, 0], 6: [0, 0, 2, 2, 0, 0]}
    passed_status = True
    for rank in [4, 5, 6]:
        algebra = semisimple_lie_algebra.A(rank)
        highest_weight_state = highest_weights[rank]
        rep = representations.Irrep(algebra, highest_weight_state)
        yt_repr_dimension = rep.dimension
        lattice = rep.weight_lattice
        lattice_repr_dimension = sum([mul for root, mul in lattice])
        passed_status &= yt_repr_dimension == lattice_repr_dimension
        if message_detail:
            _make_message(algebra.name + " ", passed_status)
            # print(yt_repr_dimension)
            # print(lattice_repr_dimension)
        if not passed_status:
            break
    _make_message("Representation total dimension test: ", passed_status)
    return passed_status
