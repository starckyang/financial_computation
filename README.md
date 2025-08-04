# Quantitative Finance Option Pricing Toolkit

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

**this repo is still under maintenance. so some of the outputs can be wrong because there are still adjustments to be made.**
**you can reach out to the other repo for correct but yet a bit messy code** : [original code](https://github.com/starckyang/financial_computation_arc/tree/main)

A Python-based **quantitative finance** toolkit that implements various **option pricing models** using both **analytical** and **numerical** methods.  
Designed for both **academic research** and **practical trading strategy development**.

---

## 📌 Features

- **Black-Scholes Model** (Analytical solution for European options)
- **Binomial Tree Model** (CRR, American, Combinatorial)
- **Asian Option Pricing** (Monte Carlo simulation)
- **Lookback Option Pricing** (Monte Carlo simulation)
- **Rainbow Option Pricing** (Monte Carlo simulation with Cholesky decomposition)
- Modular design for easy extension and maintenance
- PrettyTable-formatted output for clean presentation

---

## 📂 File Structure

```bash
│
├── main.py # Main script to run all models and compare results
│
├── models/
│ ├── black_scholes.py # Black-Scholes model
│ ├── binomial_tree.py # Binomial tree model
│ ├── asian_option.py # Asian option pricing
│ ├── lookback_option.py # Lookback option pricing
│ ├── rainbow_option.py # Rainbow option pricing
│
├── utils/
│ ├── option_math.py # Common math functions (d1/d2, combinatorics, etc.)
│ ├── tree_builder.py # Stock price tree generation helpers
│ ├── monte_carlo.py # Generic Monte Carlo simulation helpers
│
├── requirements.txt # Python dependencies
└── README.md

---
```

## 🔧 Installation

```bash
git clone https://github.com/yourusername/quant-option-toolkit.git
cd quant-option-toolkit
pip install -r requirements.txt
```

## 🚀 Usage

```python
main.py
```

You can edit main.py to change parameters such as:

```python
S0 = 100      # Spot price
K = 100       # Strike price
T = 1         # Maturity (in years)
sigma = 0.2   # Volatility
r = 0.05      # Risk-free rate
q = 0.0       # Dividend yield
n = 100       # Binomial tree steps
N_SIM = 10000 # Monte Carlo simulations
B = 5         # Monte Carlo batches
```

## 📊 Example Output
```mathematica
=== Option Pricing Results ===
+------------------------+--------------------+--------------------+
|         Model          |     Call Price     |     Put Price      |
+------------------------+--------------------+--------------------+
|     Black-Scholes      |      19.8067       |      14.9296       |
|     European (MC)      | (19.0504, 20.2957) | (14.7241, 15.2954) |
| Binomial Tree (CRR EU) |      19.7634       |      14.8863       |
| Binomial Tree (CRR AM) |      19.7634       |      15.4321       |
|  Binomial Tree (Comb)  |      19.7634       |      14.8863       |
|   Asian Option (MC)    | (-4.2832, 29.6134) |                    |
|  Lookback Option (MC)  |       36.526       |      31.8767       |
|  Rainbow Option (MC)   | (20.7441, 20.8971) |                    |
+------------------------+--------------------+--------------------+

```

## 📚 Academic & Practical Use Cases
- Coursework & Research: Financial computation, derivatives pricing, stochastic processes
- Trading Strategy Prototyping: Quick testing of option strategies with realistic assumptions
- Interview Preparation: Demonstrate understanding of both theory & coding

## 📜 License
MIT License.
Feel free to use, modify, and distribute with attribution.

## 👨‍💻 Author
Developed by Ling-Chen (Starck) Yang.
Portfolio: https://github.com/starckyang
Email: starck.catwarrior@gmail.com 
