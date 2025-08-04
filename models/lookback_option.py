"""
Lookback option pricing using Monte Carlo simulation.
Supports both Call and Put (Floating Strike).
"""

import numpy as np
from utils.monte_carlo import mc_confidence_interval

def lookback_option_main(S0, r, q, sigma, T, n=100, N_SIM=5000, B=5):
    """
    Monte Carlo pricing for floating strike lookback options.

    Parameters:
        S0 (float): Initial stock price
        r (float): Risk-free rate
        q (float): Dividend yield
        sigma (float): Volatility
        T (float): Time to maturity (years)
        n (int): Number of time steps
        N_SIM (int): Number of simulations
        B (int): Number of batches for variance reduction

    Returns:
        dict: {
            "call_price": float,
            "call_ci": (low, high),
            "put_price": float,
            "put_ci": (low, high)
        }
    """
    dt = T / n
    call_prices = []
    put_prices = []

    for _ in range(B):
        # 初始化
        St = np.full(N_SIM, float(S0))
        max_record = np.full(N_SIM, float(S0))
        min_record = np.full(N_SIM, float(S0))

        # 模擬股價路徑
        for _ in range(n):
            Z = np.random.normal(0, 1, N_SIM)
            St *= np.exp((r - q - 0.5 * sigma ** 2) * dt +
                         sigma * np.sqrt(dt) * Z)
            max_record = np.maximum(max_record, St)
            min_record = np.minimum(min_record, St)

        # 浮動行權價 payoff
        call_payoff = np.maximum(max_record - St, 0.0)
        put_payoff = np.maximum(St - min_record, 0.0)

        # 修正 — 防止 min_record 偏低（數值誤差補正）
        # 最小價格不應該低於 S0 * exp(-kσ√T) 合理區間
        lower_bound = S0 * np.exp(-5 * sigma * np.sqrt(T))
        min_record = np.maximum(min_record, lower_bound)
        put_payoff = np.maximum(St - min_record, 0.0)

        # 折現
        call_prices.append(np.mean(call_payoff) * np.exp(-r * T))
        put_prices.append(np.mean(put_payoff) * np.exp(-r * T))

    # 平均價格
    call_price = np.mean(call_prices)
    put_price = np.mean(put_prices)

    # 置信區間
    call_ci = mc_confidence_interval(call_prices)
    put_ci = mc_confidence_interval(put_prices)

    return {
        "call_price": round(call_price, 4),
        "call_ci": call_ci,
        "put_price": round(put_price, 4),
        "put_ci": put_ci
    }
