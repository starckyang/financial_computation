"""
Black-Scholes model with Monte Carlo simulation.
Supports European call/put option pricing.
"""

from utils.option_math import bs_call, bs_put
from utils.monte_carlo import mc_confidence_interval
import numpy as np

def black_scholes_main(S0, K, r, q, sigma, T, N_SIM=10000, B=10):
    # Closed-form BS prices
    bs_call_price = bs_call(S0, K, r, q, sigma, T)
    bs_put_price = bs_put(S0, K, r, q, sigma, T)

    # Monte Carlo simulation
    call_results, put_results = [], []
    collection_st = []
    for _ in range(B):
        Z = np.random.standard_normal(N_SIM)
        ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)
        # collection_st.append(np.average(ST))
        call_payoff = np.maximum(ST - K, 0)
        put_payoff = np.maximum(K - ST, 0)
        call_results.append(np.exp(-r * T) * np.mean(call_payoff))
        put_results.append(np.exp(-r * T) * np.mean(put_payoff))

    call_ci = mc_confidence_interval(call_results)
    put_ci = mc_confidence_interval(put_results)
    # print(np.mean(collection_st), np.var(collection_st))
    return {
        "BS_Call": round(bs_call_price, 4),
        "BS_Put": round(bs_put_price, 4),
        "MC_Call_CI": call_ci,
        "MC_Put_CI": put_ci
    }
