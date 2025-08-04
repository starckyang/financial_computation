"""
Rainbow option pricing with Monte Carlo simulation.
Uses Cholesky decomposition for correlated asset returns.
"""

import numpy as np
from utils.option_math import cholesky_decomposition
from utils.monte_carlo import mc_confidence_interval, antithetic_moment_matching

def rainbow_option_main(S0, var_mat, K, r, T, N_SIM=10000, B=5, n_assets=5):
    S0 = np.array(S0, dtype=float)
    var_mat = np.array(var_mat, dtype=float)
    A0 = cholesky_decomposition(np.array(var_mat))
    sim_avg = []

    for _ in range(B):
        Z = np.random.normal(0, 1, (N_SIM, n_assets))
        Z = antithetic_moment_matching(Z)  # Antithetic + Moment Matching
        data = Z.dot(A0)
        E_ST = S0 * np.exp(r * T)
        ST = data + E_ST - K
        ST = np.hstack((ST, np.zeros((ST.shape[0], 1))))  # for max calculation
        max_values = np.max(ST, axis=1)
        sim_avg.append(np.mean(max_values))

    ci = mc_confidence_interval(sim_avg)
    return {"Rainbow_Option_CI": ci}
