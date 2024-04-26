import numpy as np
from prettytable import PrettyTable


class RainbowCalculator:

    def __init__(self):
        self.S0 = np.array([
            172, 173, 169, 165, 170
        ])
        self.var_mat = np.array([
            [0.3, 0.25, 0.1, 0.2, 0.15],
            [0.25, 0.4, 0.22, 0.25, 0.08],
            [0.1, 0.22, 0.8, 0.29, 0.08],
            [0.2, 0.25, 0.29, 0.6, 0.17],
            [0.15, 0.08, 0.08, 0.17, 0.5]
        ])
        self.K, self.r, self.T, self.N_SIM, self.B, self.n = 140, 0.02, 1, 100, 5, 5
        self.A0 = self.chlo_dec(self.var_mat)
        self.simple_dgp = self.rainbow_option_pricing("normal")
        self.ant_dgp = self.rainbow_option_pricing("ant")
        self.inv_dgp = self.rainbow_option_pricing("inv_chl")

    def __str__(self):
        return "This is a calculator for rainbow options with different types of data generating processes"

    def __repr__(self):
        return "The calculator now has three types of DGPs, including simple Monte Carlo, Antithetic and Moments matching," \
               "and inverted Chloesky data processing method."

    def chlo_dec(self, matrix):
        size = matrix.shape[0]
        A = np.zeros([size, size])
        # step1
        for i in range(size):
            if i == 0:
                A[i, 0] = matrix[0, 0] ** (1/2)
            else:
                A[i, 0] = matrix[i, 0] / A[0, 0]
        # step2&3
        for i in range(size-1):
            if i != 0:
                d_sum = np.sum(A[i, :i] * A[i, :i])
                A[i, i] = (matrix[i, i] - d_sum) ** (1/2)
                for k in range(i):
                    d_sum2 = np.sum(A[i, :i] * A[k, :i])
                    A[k, i] = A[i, i] ** (-1) * (matrix[k, i] - d_sum2)
        d_sum3 = np.sum(A[size - 1, :size] * A[size - 1, :size])
        A[size - 1, size - 1] = (matrix[size - 1, size - 1] - d_sum3) ** (1 / 2)
        return A.T


    def ant_mom_data(self):
        ori_sim = np.random.normal(0, 1, (self.N_SIM, self.n))
        # antithetic approach
        neg_sim = -ori_sim
        anti_sim = np.vstack((ori_sim, neg_sim))
        # moment matching
        for i in range(self.n):
            cur_std = np.std(anti_sim[:, i])
            anti_sim[:, i] = anti_sim[:, i] / cur_std
        return anti_sim


    def inv_chl_data(self):
        ori_sim = np.random.normal(0, 1, (self.N_SIM, self.n))
        cov_mat = np.cov(ori_sim)
        cov_A = np.linalg.inv(self.chlo_dec(cov_mat))
        return cov_A.dot(ori_sim)

    def sim_calculation(self, data, n_rows):
        data = data.dot(self.A0)
        E_ST = self.S0 * np.exp(self.r * self.T)
        ST = data + E_ST - self.K
        zero_column = np.zeros((n_rows, 1))
        ST = np.hstack((ST, zero_column))
        max_values = np.amax(ST, axis=1)
        max_values = max_values.reshape(n_rows, 1)
        return np.mean(max_values)

    def rainbow_option_pricing(self, DGP):

        sim_avg = np.array([])
        for _ in range(self.B):
            if DGP == "normal" or DGP == "inv_chl":
                if DGP == "normal":
                    data = np.random.normal(0, 1, (self.N_SIM, self.n))
                else:
                    data = self.inv_chl_data()
                avg_val = self.sim_calculation(data, self.N_SIM)
                sim_avg = np.append(sim_avg, avg_val)
            elif DGP == "ant":
                data = self.ant_mom_data()
                avg_val = self.sim_calculation(data, self.N_SIM*2)
                sim_avg = np.append(sim_avg, avg_val)
        final_mean = np.mean(sim_avg)
        final_std = np.std(sim_avg)
        return (round(final_mean - 1.96 * final_std, 4), round(final_mean + 1.96 * final_std, 4))

    def outputing(self):
        output_table = PrettyTable()
        output_table.field_names = ["Raw Data", "Antithetical and MoM", "Inverted Chloesky"]
        output_table.add_row(
            [self.simple_dgp, self.ant_dgp, self.inv_dgp]
        )
        print("Your Outputs are:\n")
        print(output_table)

if __name__ == "__main__":
    rainbow_pricing = RainbowCalculator()
    rainbow_pricing.outputing()



