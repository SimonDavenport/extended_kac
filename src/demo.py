## This file runs some demos for the Lie Algebra interface

import unittest
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import src.utils.logging
from src.algebra import semisimple_lie_algebra

if __name__ == '__main__':

    src.utils.logging.setup_logger(file_name='demo')
    log = logging.getLogger('logger')

    algebra = semisimple_lie_algebra.build('F', 4)

    roots = algebra.root_system

   #lattice = algebra.root_lattice
