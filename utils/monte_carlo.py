"""
Monte Carlo helper functions for option pricing.
"""

import numpy as np


def mc_confidence_interval(values):
    """Return 95% confidence interval for Monte Carlo results."""
    mean = np.mean(values)
    std = np.std(values)
    return round(mean - 1.96 * std, 4), round(mean + 1.96 * std, 4)


def antithetic_moment_matching(z):
    """
    Apply Antithetic Variates + Moment Matching to a set of standard normal draws.
    z: np.ndarray of shape (N, steps)
    """
    z_antithetic = -z
    sims = np.vstack((z, z_antithetic))
    # Moment matching (normalize variance to 1)
    for i in range(sims.shape[1]):
        sims[:, i] /= np.std(sims[:, i])
    return sims
