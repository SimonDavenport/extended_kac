"""This file implements utility function to enumerate Young tableaux"""

import numpy as np
from src.algebra import utils


def highest_weight_to_partition(highest_weight):
    """Map the highest weight state to an integer partition, which defines
    an associated Young tableaux shape"""
    full_partition = np.cumsum(utils.round(np.flip(highest_weight, 0)))
    return np.flip(full_partition[full_partition > 0], 0)


def _hook_lengths_as_tableaux(shape):
    """Generates a Young frame containing the Hook lengths of each cell, defined
    by h(i, j) = nbr cell below + nbr cells right + 1"""
    right_counts = _to_right_counts(shape)
    transposed_right_counts = _to_right_counts(_transpose_shape(shape))
    below_counts = _transpose_tableaux(transposed_right_counts)
    return [below_count + right_count + 1 for below_count, right_count in zip(right_counts, below_counts)]


def _contents_as_tableaux(shape):
    """Generates a Young frame containing the "content" of each cell, defined
    by c(i, j) = i - j"""
    row_indexes = _to_row_indexes(shape)
    transposed_col_indexes = _to_row_indexes(_transpose_shape(shape))
    col_indexes = _transpose_tableaux(transposed_col_indexes)
    return [row_index - col_index for row_index, col_index in zip(row_indexes, col_indexes)]


def _transpose_shape(shape):
    """Generate the transposed Young frame shape"""
    nbr_cols = shape[0]
    return [sum(utils.itype(shape > shift)) for shift in range(nbr_cols)]


def _transpose_tableaux(tableaux):
    """Generate the transpose of the Young tableaux"""
    nbr_cols = len(tableaux[0])
    return [np.array([row[i] for row in tableaux if i < len(row)]) for i in range(nbr_cols)]


def _to_right_counts(shape):
    """Return a Young tableaux containing the number of cells to the right of each cell"""
    return [row_length - np.arange(row_length, dtype=utils.itype) - 1 for row_length in shape]


def _to_row_indexes(shape):
    """Return a Young tableaux containing the row index of each cell"""
    return [np.arange(row_length, dtype=utils.itype) for row_length in shape]


def _integer_product_divide(numerator, denominator, switch_criterion=10):
    """Given lists of integer numerator and denominator factors, cancel
    out some of those factors to perform an integer division"""

    def list_to_map(x):
        map_x = {}
        for element in x:
            if element > 1:
                if element in map_x.keys():
                    map_x[element] += 1
                else:
                    map_x[element] = 1
        return map_x

    def map_to_list(map_x):
        x = []
        for element in map_x.keys():
            x += [element] * map_x[element]
        return x

    if len(denominator) > switch_criterion or len(denominator) > switch_criterion:
        numerator_map = list_to_map(numerator)
        denominator_map = list_to_map(denominator)
        for factor in denominator_map:
            if factor in numerator_map.keys():
                cancel_count = min(denominator_map[factor], numerator_map[factor])
                numerator_map[factor] -= cancel_count
                denominator_map[factor] -= cancel_count
        cancelled_numerator = map_to_list(numerator_map)
        cancelled_denominator = map_to_list(denominator_map)
        return np.int64(np.prod(cancelled_numerator) / np.prod(cancelled_denominator))
    else:
        return np.int64(np.prod(numerator) / np.prod(denominator))


def _kostka_number(shape, weight):
    """The Kostka number is the number of semistandard tableaux of a given shape
    and a given weight (set of entries). """
    raise NotImplementedError("Kostka numbers are not implemented")


def tensor_product(frame1, frame2):
    """Generates a decomposition of the tensor product of two given Young frames
    via the Littlewood-Richardson rule"""
    raise NotImplementedError("Kostka numbers are not implemented")
    tensor_product_coefficients = []
    frames = []
    return tensor_product_coefficients, frames


class Frame:

    def __init__(self, shape):
        self.__shape = np.array(shape, dtype=utils.itype)
        self._check_shape()

    def _check_shape(self):
        """Check that the tableaux shape is properly defined"""
        if not np.all(np.diff(self.shape) <= 0):
            raise RuntimeError("Invalid frame shape ", self.shape)

    @property
    def shape(self):
        """Return the frame shape"""
        return self.__shape

    @property
    def nbr_cells(self):
        """Return the number of cells in the Young frame"""
        return sum(self.__shape)

    def count_standard_tableaux(self):
        """Compute and return the number of standard Young tableaux
        associated with a given Young frame"""
        hook_lengths = np.concatenate(_hook_lengths_as_tableaux(self.shape))
        factorial_factors = 1 + np.arange(self.nbr_cells)
        return _integer_product_divide(factorial_factors, hook_lengths)

    def count_total_semistandard_tableaux(self, nbr_labels):
        """Compute and return the total number of semistandrad Young tableaux
        associated with the Young frame and a given number of labels."""
        hook_lengths = np.concatenate(_hook_lengths_as_tableaux(self.shape))
        contents = np.concatenate(_contents_as_tableaux(self.shape))
        return _integer_product_divide(nbr_labels + contents, hook_lengths)

    def count_semistandard_tableaux(self, weight):
        """Compute and return the number of semistandard Young tableaux
        associated with the Young frame and a given partition of entry labels,
        called the weight. The result is equivalent to the Kostka number"""
        weight_shape = [len(row) for row in weight]
        if weight_shape != self.shape:
            raise RuntimeError("Invalid shape of weights")
        return _kostka_number(self.shape, weight)
