## This file runs some demos for the Lie Algebra interface

import unittest
import os
import sys
import logging
import numpy as np
import pickle

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.utils.logging
from src.algebra import semisimple_lie_algebra, utils, roots


if __name__ == '__main__':

    src.utils.logging.setup_logger(file_name='demo')
    log = logging.getLogger('logger')

    '''
    # E type
    algebra = semisimple_lie_algebra.build('E', 8)

    # E6
    #basis_change_matrix = np.matrix([[0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0, 0], 
    #                                    [0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 1, 1, 0], [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]])

    #E7
    #basis_change_matrix = np.matrix([[0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0, 0], 
    #                                    [0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 1, 1, 0], [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]])

    #E8
    basis_change_matrix = np.matrix([[1, -1, 0, 0, 0, 0, 0, 0], [0, 1, -1, 0, 0, 0, 0, 0], [0, 0, 1, -1, 0, 0, 0, 0], [0, 0, 0, 1, -1, 0, 0, 0], [0, 0, 0, 0, 1, -1, 0, 0], 
                                        [0, 0, 0, 0, 0, 1, -1, 0], [0, 0, 0, 0, 0, 1, 1, 0], [-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]])

    new_basis_cartan = np.dot(np.linalg.pinv(basis_change_matrix), algebra.cartan_matrix)
    
    diff = new_basis_cartan - algebra.modified_cartan_matrix
    test = np.all(abs(diff)<1e-14)
    
    # F type
    algebra = semisimple_lie_algebra.build('F', 4)

    basis_change_matrix = np.matrix([[1, -1, 0, 0], [0, 1, -1, 0], [0, 0, 1, 0], [-0.5, -0.5, -0.5, 0.5]])

    new_basis_cartan = np.dot(np.linalg.pinv(basis_change_matrix), algebra.cartan_matrix)

    diff = new_basis_cartan - algebra.modified_cartan_matrix
    test = np.all(abs(diff)<1e-14)
    '''

    '''
    algebra = semisimple_lie_algebra.build('E', 8)
    algebra1 = semisimple_lie_algebra.build('E', 8)
    
    roots = set([root.tostring() for root in algebra.root_system])

    #roots1 = pickle.load(file=open('d4_roots.pkl', 'rb'))

    algebra1.override_simplified_formula = True

    roots1 = set([root.tostring() for root in algebra1.root_system])

    #pickle.dump(roots1, file=open('e7_roots.pkl', 'wb'))

    test = roots == roots1

    diff = [utils.unserialize(root) for root in roots.difference(roots1)]
    diff1 = [utils.unserialize(root) for root in roots1.difference(roots)]

    lattice = algebra.root_lattice
    '''
