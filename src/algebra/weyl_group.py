"""This file contains an implementation of an algroithm to generate the
elements of the Weyl group associated with a given Cartan matrix"""

import logging
import numpy as np
from src.algebra import roots, utils, cartan

log = logging.getLogger('logger')

_separator = ','

def _weyl_reflection_matrix(root_index, A):
    """Construct a matrix representation for the Weyl reflection acting
    on a simple root. Definition: s_i alpha_j = alpha_j - A_ji alpha_i"""
    rank = A.shape[0]
    reflection_matrix = np.array(np.diag([1] * rank), dtype=utils.itype)
    if root_index > 0:
        reflection_matrix[:, root_index - 1] -= A[:, root_index - 1]
    return reflection_matrix


def _init_reflection_chain(A):
    """Initialize the chain of Weyl reflections using Cartan matrix data"""
    simple_reflections = {}
    weyl_reflections = set()
    weyl_elements = []
    rank = A.shape[0]
    weyl_reflections.add(utils.serialize(_weyl_reflection_matrix(0, A)))
    weyl_elements.append('')
    for r in range(1, 1 + rank):
        reflection_matrix = _weyl_reflection_matrix(r, A)
        key = str(r)
        simple_reflections[key] = reflection_matrix
        weyl_reflections.add(utils.serialize(reflection_matrix))
        weyl_elements.append(key)
    return simple_reflections, weyl_reflections, weyl_elements


def _extend_reflection_chain(current_reflections, added_reflections,
                             new_reflections, weyl_reflections, weyl_elements):
    """Test extending the chain of Weyl reflections by adding new reflections"""
    for curr_key in current_reflections.keys():
        for added_key in added_reflections.keys():
            proposed_key = curr_key + _separator + added_key
            new_reflection_matrix = np.dot(current_reflections[curr_key], added_reflections[added_key])
            new_reflection_str = utils.serialize(new_reflection_matrix)
            if new_reflection_str not in weyl_reflections:
                new_reflections[proposed_key] = new_reflection_matrix
                weyl_reflections.add(new_reflection_str)
                weyl_elements.append(proposed_key)


def generate_elements(A, serialized=False):
    """Generate the elements and reflection matrtix representations of the
    Weyl group given a Cartan matrix A"""
    log.debug("Generate Weyl group elements, serialized=" + str(serialized))
    simple_reflections, weyl_reflections, weyl_elements = _init_reflection_chain(A)
    prev_reflections = simple_reflections.copy()
    reflection_chain_iteration_count = 0
    while len(prev_reflections) > 0:
        new_reflections = {}
        _extend_reflection_chain(prev_reflections, simple_reflections,
                                 new_reflections, weyl_reflections, weyl_elements)
        _extend_reflection_chain(simple_reflections, prev_reflections,
                                 new_reflections, weyl_reflections, weyl_elements)
        prev_reflections = new_reflections.copy()
        reflection_chain_iteration_count += 1
    log.debug("Weyl group contains " + str(len(weyl_elements)) + " elements")
    log.debug("Reflection composition method completed in " + str(reflection_chain_iteration_count) + " iterations")

    if serialized:
        return [[utils.serialize(term) for term in utils.itype(weyl.split(_separator))-1] if len(weyl) > 0 else []
                for weyl in weyl_elements]
    else:
        return weyl_elements

def _weyl_reflection(beta, reflection_label, A, F, cache):
    """Perform a Weyl reflection on the given root beta. Alpha is a simple root.
        Definition: s_alpha beta = beta - (coroot(alpha), beta) alpha"""
    key = reflection_label + utils.serialize(beta)
    if key in cache:
       return cache[key]
    else:
        reflection_index = utils.unserialize(reflection_label)[0]
        alpha = A[reflection_index, :]
        refl = beta - utils.round(roots.inner_product(roots.coroot(alpha, F), F, beta)) * alpha
        cache[key] = refl
        return refl


def _element_to_reflection(serialized_weyl_element, weight, A, F, cache):
    """Convert a Weyl group element into a series of Weyl reflections acting on
    the given weight. Return the output of applying those reflections."""
    for reflection_label in serialized_weyl_element:
        weight = _weyl_reflection(weight, reflection_label, A, F, cache)
    return weight


def orbit(weight, A, F, serialized_weyl_elements=None, cache=None):
    """Compute the weight belonging to the Weyl orbit of a given root"""
    if serialized_weyl_elements is None:
        serialized_weyl_elements = generate_elements(A, serialized=True)
    if cache is None:
        cache = {}
    rank = len(weight)
    weyl_order = len(serialized_weyl_elements)
    orbit_weights = np.zeros(shape=(weyl_order, rank), dtype=utils.itype)
    for i, element in enumerate(serialized_weyl_elements):
        orbit_weights[i] = _element_to_reflection(element, weight, A, F, cache)
    return orbit_weights


def length(weyl_element):
    """The length of a Weyl group element is the minimum number of Weyl reflections that
    need to be composed in order to represent it"""
    return len(weyl_element)


def signature(weyl_element):
    """The signature of a Weyl group elemenet, denoted by epsilon(w) is (-1)**length(w)"""
    return (-1) ** length(weyl_element)


def generate_root_system(A, F, simple_root_basis=False):
    """Generate the root system using the action of the elements of the Weyl group
    on the simple roots."""
    log.debug("Generate root system, simple_root_basis=" + str(simple_root_basis))
    roots = set()
    serialized_weyl_elements = generate_elements(A, serialized=True)
    cache = {}
    for simple_root in A:
        orbit_roots = orbit(simple_root, A, F, serialized_weyl_elements, cache)
        for root in orbit_roots:
            roots.add(utils.serialize(root))
    log.debug("Root sytem contains " + str(len(roots)) + " roots")
    if simple_root_basis:
        return np.array([roots.to_simple_root_basis(utils.unserialize(root), A) for root in roots])
    else:
        return np.array([utils.unserialize(root) for root in roots], dtype=utils.itype)
