from scipy.stats import norm
import numpy as np
from prettytable import PrettyTable
import os

# inputs

S0, K, T, sigma, r, q, N_SIM, B, n = [float(data) for data in input("please insert in the following format:"
                                                                    "\nS0, K, T, sigma, r, q, N_SIM, B, n\n").split(",")]


# BS


def d1_c(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r -q + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def d2_c(S0, K, r, q, sigma, T):
    return (np.log(S0 / K) + (r - q - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))


def bs_call(S0, K, r, q, sigma, T):
    return S0 * np.exp(-q * T) * norm.cdf(d1_c(S0, K, r, q, sigma, T)) - K * np.exp(-r * T) * norm.cdf(d2_c(S0, K, r, q, sigma, T))


def bs_put(S0, K, r, q, sigma, T):
    return K * np.exp(-r * T) * norm.cdf(-d2_c(S0, K, r, q, sigma, T)) - S0 * np.exp(-q * T) * norm.cdf(-d1_c(S0, K, r, q, sigma, T))

bs_call_ans = bs_call(S0, K, r, q, sigma, T)
bs_put_ans = bs_put(S0, K, r, q, sigma, T)
print("BS formula: DONE")

# monte carlo


num_simulations = int(N_SIM)  # Number of simulated asset paths
op_prices_call = np.array([])
op_prices_put = np.array([])
for i in range(int(B)):
    # Monte Carlo simulation of asset paths
    Z = np.random.standard_normal(num_simulations)
    ST = S0 * np.exp((r - q - 0.5 * sigma ** 2) * T + sigma * np.sqrt(T) * Z)

    # Payoff calculation for each simulation
    payoff_call = np.piecewise(ST,
                          [ST > K, ST <= K],
                          [lambda ST: ST-K, 0])

    payoff_put = np.piecewise(ST,
                               [ST < K, ST >= K],
                               [lambda ST: K - ST, 0])

    # Discount the expected payoff to present value
    option_price_mc_call = np.exp(-r * T) * np.mean(payoff_call)
    op_prices_call = np.append(op_prices_call, option_price_mc_call)
    option_price_mc_put = np.exp(-r * T) * np.mean(payoff_put)
    op_prices_put = np.append(op_prices_put, option_price_mc_put)

call_mean, call_sd = op_prices_call.mean(), op_prices_call.std()
put_mean, put_sd = op_prices_put.mean(), op_prices_put.std()
call_ci_mc = (call_mean - 2 * call_sd, call_mean + 2 * call_sd)
put_ci_mc = (put_mean - 2 * put_sd, put_mean + 2 * put_sd)
print("Monte Carlo Simulation: DONE")

# CRR BTM

n = int(n)
M = np.zeros((n+1, n+1))
M[0, 0] = S0
u = np.exp(sigma*np.sqrt(T/n))
d = np.exp(-sigma*np.sqrt(T/n))
p = (np.exp(r*T/n) - d) / (u-d)
q_ = 1 - p
for i in range(n):
    M[:, i+1] = M[:, i] * u
    M[i+1, i+1] = M[i, i] * d
M_c = np.zeros((n+1, n+1))
M_p = np.zeros((n+1, n+1))
M_ca = np.zeros((n+1, n+1))
M_pa = np.zeros((n+1, n+1))
ST = M[:, -1]
M_c[:, -1] = np.piecewise(ST,
                        [ST > K, ST <= K],
                        [lambda ST: ST-K, 0])
M_p[:, -1] = np.piecewise(ST,
                        [ST < K, ST >= K],
                        [lambda ST: K-ST, 0])

for i in range(n):
    M_c[:n-i, n-1-i] = (M_c[:n-i, (n-i)] * p + M_c[1:n+1-i, n-i] * q_) * np.exp(-r*T/n)
    M_p[:n-i, n-1-i] = (M_p[:n-i, (n-i)] * p + M_p[1:n+1-i, n-i] * q_) * np.exp(-r*T/n)
CRR_call_ans = M_c[0, 0]
CRR_put_ans = M_p[0, 0]
print("Matrix CRR: DONE")

# CRR with one column

M = np.zeros((n+1, 1))
M[0, 0] = S0
for i in range(n):
    M[i+1] = M[i] * d
    M[:i+1] = M[:i+1] * u
M_c = np.piecewise(M,
                   [M > K, M <= K],
                   [lambda M: M-K, 0])
M_p = np.piecewise(M,
                   [M < K, M >= K],
                   [lambda M: K-M, 0])
for i in range(n):
    M_c[:n-i] = (M_c[:n-i] * p + M_c[1:n+1-i] * q_) * np.exp(-r*T/n)
    M_p[:n-i] = (M_p[:n-i] * p + M_p[1:n+1-i] * q_) * np.exp(-r*T/n)

OC_CRR_call_ans = M_c[0, 0]
OC_CRR_put_ans = M_p[0, 0]
print("Vector CRR: DONE")


# combinatorial method


def log_factorial(n, p):
    total = 0
    if (n == p) or (p == 0):
        return 1
    for i in range(min(p, n-p)):
        total += np.log(n-i)
        total -= np.log(i+1)
    return total


call_price = 0
put_price = 0
for i in range(n+1):
    likelihood = np.exp(log_factorial(n, i) + (n-i) * np.log(p) + i * np.log(q_))
    if S0 * (u ** (n-i)) * (d ** i) > K:
        call_price += (S0 * (u ** (n-i)) * (d ** i) - K) * likelihood
    else:
        put_price += (K - S0 * (u ** (n-i)) * (d ** i)) * likelihood
comb_call_price = call_price * np.exp(-r * T)
comb_put_price = put_price * np.exp(-r * T)

print("Combinatorial Method: DONE")
os.system("cls" if os.name == 'nt' else 'clear')

input_table = PrettyTable()
input_table.field_names = ['S0', 'K', 'T', 'sigma', 'r', 'q', 'N_SIM', 'B', 'n']
input_table.add_row([S0, K, T, sigma, r, q, N_SIM, B, n])

output_table = PrettyTable()
output_table.field_names = ["P/C", "BS formula", "MC sim", "CRR_matrix", "CRR_vector", "Combinatorial"]
output_table.add_rows(
    [["Call", bs_call_ans, call_ci_mc, CRR_call_ans, OC_CRR_call_ans, comb_call_price],
    ["Put", bs_put_ans, put_ci_mc, CRR_put_ans, OC_CRR_put_ans, comb_put_price]]
)

print("Your Inputs are:\n")
print(input_table)
print("Your Outputs are:\n")
print(output_table)

