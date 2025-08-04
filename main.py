"""
main.py
Run multiple option pricing models and compare results.
"""

from prettytable import PrettyTable
from models.black_scholes import black_scholes_main
from models.binomial_tree import binomial_tree_main
from models.asian_option import asian_option_main
from models.lookback_option import lookback_option_main
from models.rainbow_option import rainbow_option_main

# =======================
# Default Input Parameters
# =======================
S0 = 100      # Spot price
K = 100       # Strike price
T = 1         # Time to maturity (in years)
sigma = 0.2   # Volatility
r = 0.05      # Risk-free rate
q = 0.0       # Dividend yield
n = 100       # Steps for binomial tree
N_SIM = 10000 # Monte Carlo simulations
B = 5         # Monte Carlo batches
S0, K, sigma, r, q, T = map(float, (S0, K, sigma, r, q, T))

# =======================
# Run Models
# =======================
print("Running Black-Scholes...")
bs_results = black_scholes_main(S0, K, r, q, sigma, T)

print("Running Binomial Tree...")
bt_results = binomial_tree_main(S0, K, r, q, sigma, T, n)

print("Running Asian Option...")
asian_results = asian_option_main(S0, K, r, q, sigma, T, N_SIM, B, n)

print("Running Lookback Option...")
lookback_results = lookback_option_main(S0, r, q, sigma, T, N_SIM, B, n)

print("Running Rainbow Option...")
rainbow_results = rainbow_option_main(
    S0=[172, 173, 169, 165, 170],
    var_mat=[
        [0.3, 0.25, 0.1, 0.2, 0.15],
        [0.25, 0.4, 0.22, 0.25, 0.08],
        [0.1, 0.22, 0.8, 0.29, 0.08],
        [0.2, 0.25, 0.29, 0.6, 0.17],
        [0.15, 0.08, 0.08, 0.17, 0.5]
    ],
    K=140, r=r, T=T, N_SIM=5000, B=5, n_assets=5
)

# =======================
# Output Results in Table
# =======================
table = PrettyTable()
table.field_names = ["Model", "Call Price", "Put Price"]

# Black-Scholes
table.add_row(["Black-Scholes", bs_results["BS_Call"], bs_results["BS_Put"]])

# Binomial Tree - CRR European
table.add_row(["Binomial Tree (CRR EU)", bt_results["CRR_Call_EU"], bt_results["CRR_Put_EU"]])

# Binomial Tree - CRR American
table.add_row(["Binomial Tree (CRR AM)", bt_results["CRR_Call_AM"], bt_results["CRR_Put_AM"]])

# Binomial Tree - Combinatorial
table.add_row(["Binomial Tree (Comb)", bt_results["Comb_Call"], bt_results["Comb_Put"]])

# Asian Option
table.add_row(["Asian Option (MC)", asian_results["Asian_Option_CI"], ""])

# Lookback Option
table.add_row(["Lookback Option (MC)", lookback_results["call_price"], lookback_results["put_price"]])

# Rainbow Option
table.add_row(["Rainbow Option (MC)", rainbow_results["Rainbow_Option_CI"], ""])

# Print
print("\n=== Option Pricing Results ===")
print(table)
