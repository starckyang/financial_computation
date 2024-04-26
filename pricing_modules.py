from scipy.stats import norm
import numpy as np
from prettytable import PrettyTable


class PricingCalculator:

    def __init__(self):
        self.S0, self.K, self.T, self.sigma, self.r, self.q, self.N_SIM, self.B, self.n = \
            [float(data) for data in input("please insert in the following format:"
                                           "\nS0, K, T, sigma, r, q, N_SIM, B, n\n").split(",")]
        self.u = np.exp(self.sigma * np.sqrt(self.T / self.n))
        self.d = np.exp(-self.sigma * np.sqrt(self.T / self.n))
        self.p = (np.exp(self.r * self.T / self.n) - self.d) / (self.u - self.d)
        self.q_ = 1 - self.p
        self.bs_call, self.bs_put = self.bs_model()
        self.monte_call, self.monte_put = self.monte_carlo()
        self.ccrr_call, self.ccrr_put = self.clean_crr()
        self.dcrr_call, self.dcrr_put = self.dirty_crr()
        self.comb_call, self.comb_put = self.combinatorial_method()

    def bs_model(self):
        def d1_c():
            return (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * self.sigma ** 2) * self.T) / (self.sigma * np.sqrt(self.T))

        def d2_c():
            return d1_c() - self.sigma * self.T**(1/2)

        def bs_call():
            return self.S0 * np.exp(-self.q * self.T) * norm.cdf(d1_c()) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2_c())

        def bs_put():
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2_c()) - self.S0 * np.exp(-self.q * self.T) * norm.cdf(-d1_c())

        return bs_call(), bs_put()

    def monte_carlo(self):
        num_simulations = int(self.N_SIM)  # Number of simulated asset paths
        op_prices_call = np.array([])
        op_prices_put = np.array([])
        for i in range(int(self.B)):
            Z = np.random.standard_normal(num_simulations)
            ST = self.S0 * np.exp((self.r - self.q - 0.5 * self.sigma ** 2) *
                                  self.T + self.sigma * np.sqrt(self.T) * Z)
            payoff_call = np.piecewise(ST,
                                  [ST > self.K, ST <= self.K],
                                  [lambda ST: ST-self.K, 0])
            payoff_put = np.piecewise(ST,
                                       [ST < self.K, ST >= self.K],
                                       [lambda ST: self.K - ST, 0])
            option_price_mc_call = np.exp(-self.r * self.T) * np.mean(payoff_call)
            op_prices_call = np.append(op_prices_call, option_price_mc_call)
            option_price_mc_put = np.exp(-self.r * self.T) * np.mean(payoff_put)
            op_prices_put = np.append(op_prices_put, option_price_mc_put)
        call_mean, call_sd = op_prices_call.mean(), op_prices_call.std()
        put_mean, put_sd = op_prices_put.mean(), op_prices_put.std()
        return (call_mean - 2 * call_sd, call_mean + 2 * call_sd), (put_mean - 2 * put_sd, put_mean + 2 * put_sd)

    def dirty_crr(self):
        self.n = int(self.n)
        M = np.zeros((self.n + 1, self.n + 1))
        M[0, 0] = self.S0
        for i in range(self.n):
            M[:, i + 1] = M[:, i] * self.u
            M[i + 1, i + 1] = M[i, i] * self.d
        M_c = np.zeros((self.n + 1, self.n + 1))
        M_p = np.zeros((self.n + 1, self.n + 1))
        ST = M[:, -1]
        M_c[:, -1] = np.piecewise(ST,
                                  [ST > self.K, ST <= self.K],
                                  [lambda ST: ST - self.K, 0])
        M_p[:, -1] = np.piecewise(ST,
                                  [ST < self.K, ST >= self.K],
                                  [lambda ST: self.K - ST, 0])

        for i in range(self.n):
            M_c[:self.n - i, self.n - 1 - i] = (M_c[:self.n - i, (self.n - i)] * self.p +
                                                M_c[1:self.n + 1 - i, self.n - i] * self.q_) \
                                               * np.exp(-self.r * self.T / self.n)
            M_p[:self.n - i, self.n - 1 - i] = (M_p[:self.n - i, (self.n - i)]
                                                * self.p + M_p[1:self.n + 1 - i, self.n - i] * self.q_) \
                                               * np.exp(-self.r * self.T / self.n)

        return M_c[0, 0], M_p[0, 0]

# CRR BTM
    def clean_crr(self):
        self.n = int(self.n)
        M = np.zeros((self.n+1, self.n+1))
        M[0, 0] = self.S0
        for i in range(self.n):
            M[i+1] = M[i] * self.d
            M[:i+1] = M[:i+1] * self.u
        M_c = np.piecewise(M,
                           [M > self.K, M <= self.K],
                           [lambda M: M-self.K, 0])
        M_p = np.piecewise(M,
                           [M < self.K, M >= self.K],
                           [lambda M: self.K-M, 0])
        for i in range(self.n):
            M_c[:self.n-i] = (M_c[:self.n-i] * self.p + M_c[1:self.n+1-i] * self.q_) * np.exp(-self.r*self.T/self.n)
            M_p[:self.n-i] = (M_p[:self.n-i] * self.p + M_p[1:self.n+1-i] * self.q_) * np.exp(-self.r*self.T/self.n)
        return M_c[0, 0], M_p[0, 0]

    def combinatorial_method(self):
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
        for i in range(self.n+1):
            likelihood = np.exp(log_factorial(self.n, i) + (self.n-i) * np.log(self.p) + i * np.log(self.q_))
            if self.S0 * (self.u ** (self.n-i)) * (self.d ** i) > self.K:
                call_price += (self.S0 * (self.u ** (self.n-i)) * (self.d ** i) - self.K) * likelihood
            else:
                put_price += (self.K - self.S0 * (self.u ** (self.n-i)) * (self.d ** i)) * likelihood
        return call_price * np.exp(-self.r * self.T), put_price * np.exp(-self.r * self.T)

    def printing(self):
        input_table = PrettyTable()
        input_table.field_names = ['S0', 'K', 'T', 'sigma', 'r', 'q', 'N_SIM', 'B', 'n']
        input_table.add_row([self.S0, self.K, self.T, self.sigma, self.r, self.q, self.N_SIM, self.B, self.n])

        output_table = PrettyTable()
        output_table.field_names = ["P/C", "BS formula", "MC sim", "CRR_matrix", "CRR_vector", "Combinatorial"]
        output_table.add_rows(
            [["Call", self.bs_call, self.monte_call, self.dcrr_call, self.ccrr_call, self.comb_call],
             ["Put", self.bs_put, self.monte_put, self.dcrr_put, self.dcrr_put, self.comb_put]]
        )
        print("Your Inputs are:\n")
        print(input_table)
        print("Your Outputs are:\n")
        print(output_table)


if __name__ == "__main__":
    calculator = PricingCalculator()
    calculator.printing()

