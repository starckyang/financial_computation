"""
Binomial Tree option pricing (CRR Model).
Supports European & American options (call & put).
Includes standard matrix version, one-column optimized version, and combinatorial method.
"""

import numpy as np
from utils.option_math import binomial_params, log_factorial


def binomial_tree_main(S0, K, r, q, sigma, T, n=100):
    # --- Binomial parameters ---
    u, d, p = binomial_params(sigma, r, q, T, n)
    q_ = 1 - p
    dt = T / n
    disc = np.exp(-r * dt)

    # === 1. Standard Matrix Version (European) ===
    stock_tree = np.zeros((n + 1, n + 1))
    stock_tree[0, 0] = S0
    for i in range(1, n + 1):
        stock_tree[i, i] = stock_tree[i - 1, i - 1] * d
        for j in range(i):
            stock_tree[j, i] = stock_tree[j, i - 1] * u

    call_tree = np.maximum(stock_tree[:, -1] - K, 0)
    put_tree = np.maximum(K - stock_tree[:, -1], 0)

    for i in range(n - 1, -1, -1):
        call_tree[:i + 1] = (p * call_tree[:i + 1] + q_ * call_tree[1:i + 2]) * disc
        put_tree[:i + 1] = (p * put_tree[:i + 1] + q_ * put_tree[1:i + 2]) * disc

    CRR_call_EU = round(call_tree[0], 4)
    CRR_put_EU = round(put_tree[0], 4)

    # === 2. American Option Pricing (Matrix) ===
    call_tree = np.maximum(stock_tree[:, -1] - K, 0)
    put_tree = np.maximum(K - stock_tree[:, -1], 0)

    for i in range(n - 1, -1, -1):
        call_cont = (p * call_tree[:i + 1] + q_ * call_tree[1:i + 2]) * disc
        put_cont = (p * put_tree[:i + 1] + q_ * put_tree[1:i + 2]) * disc

        call_tree[:i + 1] = np.maximum(stock_tree[:i + 1, i] - K, call_cont)
        put_tree[:i + 1] = np.maximum(K - stock_tree[:i + 1, i], put_cont)

    CRR_call_AM = round(call_tree[0], 4)
    CRR_put_AM = round(put_tree[0], 4)

    # === 3. One-Column Optimized Version ===
    stock_prices = np.zeros(n + 1)
    stock_prices[0] = S0
    for i in range(1, n + 1):
        stock_prices[i] = stock_prices[i - 1] * d
        stock_prices[:i] *= u

    call_vals = np.maximum(stock_prices - K, 0)
    put_vals = np.maximum(K - stock_prices, 0)

    for i in range(n - 1, -1, -1):
        call_vals[:i + 1] = (p * call_vals[1:i + 2] + q_ * call_vals[:i + 1]) * disc
        put_vals[:i + 1] = (p * put_vals[1:i + 2] + q_ * put_vals[:i + 1]) * disc

    OC_call_EU = round(call_vals[0], 4)
    OC_put_EU = round(put_vals[0], 4)

    # === 4. American One-Column Version ===
    stock_prices[0] = S0
    for i in range(1, n + 1):
        stock_prices[i] = stock_prices[i - 1] * d
        stock_prices[:i] *= u

    call_vals = np.maximum(stock_prices - K, 0)
    put_vals = np.maximum(K - stock_prices, 0)

    for i in range(n - 1, -1, -1):
        call_cont = (p * call_vals[1:i + 2] + q_ * call_vals[:i + 1]) * disc
        put_cont = (p * put_vals[1:i + 2] + q_ * put_vals[:i + 1]) * disc

        call_vals[:i + 1] = np.maximum(stock_prices[:i + 1] - K, call_cont)
        put_vals[:i + 1] = np.maximum(K - stock_prices[:i + 1], put_cont)

    OC_call_AM = round(call_vals[0], 4)
    OC_put_AM = round(put_vals[0], 4)

    # === 5. Combinatorial Method ===
    comb_call = 0
    comb_put = 0
    for i in range(n + 1):
        likelihood = np.exp(log_factorial(n, i) + (n - i) * np.log(p) + i * np.log(q_))
        ST = S0 * (u ** (n - i)) * (d ** i)
        if ST > K:
            comb_call += (ST - K) * likelihood
        else:
            comb_put += (K - ST) * likelihood

    comb_call = round(comb_call * np.exp(-r * T), 4)
    comb_put = round(comb_put * np.exp(-r * T), 4)

    return {
        "CRR_Call_EU": CRR_call_EU,
        "CRR_Put_EU": CRR_put_EU,
        "CRR_Call_AM": CRR_call_AM,
        "CRR_Put_AM": CRR_put_AM,
        "OC_Call_EU": OC_call_EU,
        "OC_Put_EU": OC_put_EU,
        "OC_Call_AM": OC_call_AM,
        "OC_Put_AM": OC_put_AM,
        "Comb_Call": comb_call,
        "Comb_Put": comb_put
    }
