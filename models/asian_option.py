"""
Asian option pricing with Monte Carlo simulation.
"""

import numpy as np
from utils.monte_carlo import mc_confidence_interval

def asian_option_main(S0, K, r, q, sigma, T, n=100, N_SIM=5000, B=5):
    dt = T / n
    mc_results = []

    for _ in range(B):
        prices = np.full((N_SIM,), float(S0))
        avg_prices = np.zeros(N_SIM)
        for _ in range(n):
            Z = np.random.normal(0, 1, N_SIM)
            prices *= np.exp((r - q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * Z)
            avg_prices += prices
        avg_prices /= n
        payoff = np.maximum(avg_prices - K, 0)
        mc_results.append(np.mean(payoff) * np.exp(-r * T))

    ci = mc_confidence_interval(mc_results)
    return {"Asian_Option_CI": ci,
            "Call": np.mean(mc_results)}
