"""
Rainbow option pricing with Monte Carlo simulation.
Uses Cholesky decomposition for correlated asset returns.
"""

import numpy as np
from utils.option_math import cholesky_decomposition
from utils.monte_carlo import mc_confidence_interval, antithetic_moment_matching

def rainbow_option_main(S0, var_mat, K, r, q, T, N_SIM=10000, B=5):
    S0 = np.array(S0, dtype=float)
    var_mat = np.array(var_mat, dtype=float)
    A0 = cholesky_decomposition(var_mat)  # covariance matrix
    sim_avg = []

    # 提取波動率（sqrt of variance）
    sigma_vec = np.sqrt(np.diag(var_mat))
    drift = (r - q - 0.5 * sigma_vec**2) * T
    collection_st = []
    for _ in range(B):
        Z = np.random.normal(0, 1, (N_SIM, len(S0)))
        Z = antithetic_moment_matching(Z)
        correlated_Z = A0.dot(Z.T).T

        # GBM 模擬
        ST = S0 * np.exp(drift + correlated_Z * np.sqrt(T))
        # collection_st.append(np.average(ST))
        payoff = np.maximum(ST - K, 0)
        best_of = np.max(payoff, axis=1)  # Best-of payoff
        sim_avg.append(np.mean(best_of))

    # print(np.mean(collection_st), np.var(collection_st))
    ci = mc_confidence_interval(sim_avg)
    return {"Rainbow_Option_CI": ci}
