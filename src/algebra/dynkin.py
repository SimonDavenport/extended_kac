"""This file contains some simple tools to draw Dynkin diagrams"""

import numpy as np
from . import cartan

___line_symbols = {1: chr(8211), 2: '=', 3: chr(8801)}
___root_symbols = {False: chr(9679), True: chr(9675)}
__filler = 2 * ' '
__vert_line = '|'


def _line_symbol(nbr_lines):
    """Return a symnol to represent 1, 2 or 3 lines"""
    return ___line_symbols[nbr_lines]


def _root_symbol(root_length):
    """Long roots: filled circles; short roots: empty circles"""
    return ___root_symbols[root_length > 1]


def _draw_spine(connecting_lines, root_ratios):
    """Draw the spine of the Dynkin diagram, a simple chain of roots"""
    spine = np.diagonal(connecting_lines, 1)
    spine_line_counts = spine[spine.nonzero()]
    nnz = len(spine_line_counts)
    if 0 == nnz:
        return _root_symbol(1)
    spine_root_ratios = root_ratios[:nnz]
    spine_root_lengths = np.cumprod(np.concatenate(([1], spine_root_ratios)))
    spine_root_lengths /= min(spine_root_lengths)
    diagram_layer = _root_symbol(spine_root_lengths[0])
    for line_count, root_length in zip(spine_line_counts, spine_root_lengths[1:]):
        diagram_layer += _line_symbol(line_count)
        diagram_layer += _root_symbol(root_length)
    return diagram_layer


def _draw_branches(connecting_lines):
    """Draw the branches of the Dynkin diagram. Root lengths are always 1
    for branches because the total number of lines to a node cannot exceed 3"""
    nbr_roots = connecting_lines.shape[0]
    diagram_filler_layer = ''
    diagram_layer = ''
    branch_root_length = 1
    is_branched = False
    for root in range(0, nbr_roots-2):
        branch_root_line_count = connecting_lines[root, -1]
        if branch_root_line_count > 0:
            diagram_filler_layer += __vert_line
            diagram_layer += _root_symbol(branch_root_length)
            is_branched = True
        else:
            diagram_filler_layer += __filler
            diagram_layer += __filler
    if is_branched:
        return "\n" + diagram_filler_layer + "\n" + diagram_layer
    else:
        return ""


def draw_diagram(A):
    """Draw Dynkin diagram given a Cartan matrix A"""
    connecting_lines = cartan.nbr_connecting_lines(A)
    root_ratios = cartan.root_ratios(A)
    diagram = ""
    diagram += _draw_spine(connecting_lines, root_ratios)
    diagram += _draw_branches(connecting_lines)
    return diagram
