"""Come utility functions for serialization and data types"""

import numpy as np

itype = np.int16


class WeylGroupError(Exception):
    def __init__(self, message):
        super(WeylGroupError, self).__init__(message)


class RankError(Exception):
    def __init__(self, message):
        super(RankError, self).__init__(message)


def serialize(data):
    """Convert given data into a serialized string"""
    return data.tostring()


def unserialize(string_data):
    """Convert a serialized string data into its original format"""
    return np.frombuffer(string_data, dtype=itype)


def round(float_data):
    """Convert a float vector to integer vector using proper rounding"""
    rounding_scale = 1.01
    return np.multiply(float_data, rounding_scale).astype(itype)


def is_integer_type_vector(vector):
    """A test to see if all elements of a vector have integer type"""
    return np.all([isinstance(val, (int, itype, np.int64)) for val in vector])


def is_integer_vector(vector):
    """A test to see if all elements of a float vector represent integers"""
    zero_tol = 1e-10
    return np.all(np.abs(vector - np.round(vector)) < zero_tol)
