"""
Mathematical formulas for option pricing models.
Includes Black-Scholes, combinatorial formulas, and Cholesky decomposition.
"""

import numpy as np
from scipy.stats import norm


# ===== Black-Scholes =====
def d1(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2(S0, K, r, q, sigma, T):
    return d1(S0, K, r, q, sigma, T) - sigma * np.sqrt(T)


def bs_call(S0, K, r, q, sigma, T):
    return S0 * np.exp(-q * T) * norm.cdf(d1(S0, K, r, q, sigma, T)) - \
           K * np.exp(-r * T) * norm.cdf(d2(S0, K, r, q, sigma, T))


def bs_put(S0, K, r, q, sigma, T):
    return K * np.exp(-r * T) * norm.cdf(-d2(S0, K, r, q, sigma, T)) - \
           S0 * np.exp(-q * T) * norm.cdf(-d1(S0, K, r, q, sigma, T))


# ===== Binomial Params =====
def binomial_params(sigma, r, q, T, n):
    dt = T / n
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    return u, d, p


# ===== Combinatorial =====
def log_factorial(n, p):
    total = 0
    if p == 0 or p == n:
        return 1
    for i in range(min(p, n - p)):
        total += np.log(n - i) - np.log(i + 1)
    return total


# ===== Cholesky =====
def cholesky_decomposition(matrix):
    size = matrix.shape[0]
    A = np.zeros([size, size])
    for i in range(size):
        for j in range(i + 1):
            sum_val = np.dot(A[i, :j], A[j, :j])
            if i == j:
                A[i, j] = np.sqrt(max(matrix[i, i] - sum_val, 0))
            else:
                A[i, j] = (matrix[i, j] - sum_val) / A[j, j]
    return A
