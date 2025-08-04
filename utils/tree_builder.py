"""
Functions to build stock price trees, max-value trees, and average trees for Binomial models.
"""

import numpy as np
import copy


def build_stock_tree(S0, u, d, n):
    """Build a recombining binomial tree for stock prices."""
    M = np.zeros((n + 1, n + 1))
    M[0, 0] = S0
    for i in range(1, n + 1):
        M[i, i] = M[i - 1, i - 1] * d
        for j in range(i):
            M[j, i] = M[j, i - 1] * u
    return M


def retain_unique(values, tolerance=1e-10):
    """Keep only unique values within tolerance."""
    unique_values = []
    for value in values:
        if (not any(abs(value - uv) < tolerance for uv in unique_values)) and (value != 0):
            unique_values.append(value)
    return unique_values


def rough_search(values, target, tolerance=1e-10):
    """Find index of value in list within tolerance."""
    for index, value in enumerate(values):
        if abs(value - target) < tolerance:
            return index
    return -1
