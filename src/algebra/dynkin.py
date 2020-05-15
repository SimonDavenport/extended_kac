"""This file contains some simple tools to draw Dynkin diagrams"""

import numpy as np
import anytree
from src.algebra import cartan

___line_symbols = {1: chr(8211), 2: '=', 3: chr(8801)}
___root_symbols = {False: chr(9679), True: chr(9675)}
__filler = 2 * ' '
__vert_line = '|'


def _line_symbol(nbr_lines):
    """Return a symnol to represent 1, 2 or 3 lines"""
    return ___line_symbols[nbr_lines]


def _root_symbol(root_length, max_root_length):
    """Long roots: filled circles; short roots: empty circles"""
    return ___root_symbols[root_length == max_root_length]


def _get_diagram_leaves_to_render(connecting_lines, root_ratios):
    """Convert adjacency matrix and root ratios data into a tree data structure
    and return a list of paths through the tree terminating in a leaf and 
    sorted by depth"""
    root_lengths = np.cumprod(np.concatenate(([1], root_ratios)))
    normalized_root_lengths = root_lengths / min(root_lengths)
    node_list = np.array([])
    for root_index, adjacency_data, root_length in zip(range(0, len(normalized_root_lengths)), 
                                                       connecting_lines, normalized_root_lengths):
        if root_index==0:
            parent_node = None
            line_symbol = None
        else:
            parent_connections = connecting_lines[root_index, :][0:root_index]
            parent_node = node_list[parent_connections.astype(bool)][0]
            line_count = parent_connections[parent_connections.astype(bool)][0]
            line_symbol = _line_symbol(line_count)
        kwargs = {'root_index': root_index, 
                  'line_symbol': line_symbol, 
                  'root_symbol': _root_symbol(root_length, max(normalized_root_lengths))}
        node_list = np.append(node_list, anytree.AnyNode(parent=parent_node, 
                                                         kwargs=kwargs))
    leaves_to_render = []
    leaf_depths = []
    for node in node_list:
        if node.is_leaf:
            leaves_to_render.append(node)
            leaf_depths.append(node.depth)
    leaves_to_render = np.array(leaves_to_render)[np.argsort(-np.array(leaf_depths))]
    return leaves_to_render


def draw_diagram(A):
    """Draw Dynkin diagram given a Cartan matrix A"""
    leaves_to_render = _get_diagram_leaves_to_render(cartan.nbr_connecting_lines(A), cartan.root_ratios(A))
    roots_rendered = []
    diagram_layers = []
    diagram_filler_layers = []
    for leaf in leaves_to_render:
        diagram_filler_layer = ''
        diagram_layer = ''
        first_in_layer = True
        for node in leaf.path:
            if node.kwargs['root_index'] not in roots_rendered:
                if first_in_layer:
                    if not node.is_root:
                        diagram_filler_layer += __filler * (node.depth-1)
                        diagram_layer += __filler * (node.depth-1)
                        diagram_filler_layer += __vert_line
                    first_in_layer = False
                else:
                    diagram_layer += node.kwargs['line_symbol']
                diagram_layer += node.kwargs['root_symbol']
                roots_rendered.append(node.kwargs['root_index'])
        diagram_layers.append(diagram_layer)
        diagram_filler_layers.append(diagram_filler_layer)

    diagram  = "\n".join([val for pair in zip(diagram_filler_layers, diagram_layers) for val in pair])

    return diagram
