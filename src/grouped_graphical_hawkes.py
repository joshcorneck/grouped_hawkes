#%%
"""
This module contains the class for simulating from the grouped Hawkes model
with an exponential kernel.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class GroupedExponentialHawkes:
    """
    A class to simulate from the grouped Hawkes model on a pre-determined graph
    structure. 
    
    Parameters:
        - G: list of group sizes (len(G) = H and sum(G) = U).
        - P: number of processes.
        - lam: a np.array of baseline intensities for each group of shape (P, H).
               This needs to be input as lam_{ph}, so that each column corresponds
               to a group and each rows to a process.
        - beta: a np.array of excitation parameters of shape (H, P, P). This needs
                to be input beta_{jph}, with column p of the h'th matrix (calling
                the tensor a stack of matrices) being the parameters corresponding 
                to excitations of process p. beta[h, j, p] = beta_{jph}
        - theta: vector of decay parameters, with the same form as beta.
        - T_max: maximum time value.
        - random_params: Boolean to indicate if we want to randomly sample
                         the parameters (just for testing purposes).
    """
    def __init__(self, G=np.array, P=2, lam=np.array, beta=np.array, 
                 theta=np.array, T_max=10, random_all=False) -> None:

        self.H = len(G); self.U = sum(G) 
        if random_all:
            lam = np.random.uniform(low=0.1, high=1.2, size=self.H * P)
            beta = np.random.uniform(low=0.2, high=1.0, size=self.H * P**2)
            theta = np.random.uniform(low=1.1, high=3.5, size=self.H * P**2)
        self.G = G; self.P = P; 
        self.lam = lam.reshape((P, self.H))
        self.beta = beta.reshape((self.H, P, P))
        self.theta = theta.reshape((self.H, P, P))
        self.T_max = T_max

    def simulate(self) -> (np.array, pd.DataFrame):
        """
        Iterate over groups and repeat for number of users in each group to generate
        an array of observations data.
        
        Returns:
            - full_data: an array with four columns, corresponding to: arrival times,
                         the process the arrival is from, the group the user belongs to,
                         the user number within that group. 
        """
        # Iterate over each group, and for the number of users wanted in each group
        full_data_list = []
        # Instantiate total user count
        u_num = -1
        for h in range(self.H):
            list_data_g = [] # Empty list to store sims for users in group h
            for u in range(self.G[h]):
                u_num += 1
                list_data_g.append(self._simulate_single_user(h, u_num))
            data_g = np.concatenate(list_data_g, axis=0)
            data_g = np.concatenate(
                [data_g, 
                np.array([h] * len(data_g)).reshape((len(data_g), 1))
                ],
                axis=1)
            full_data_list.append(data_g)

        full_data = np.concatenate(full_data_list, axis=0)

        df_full_data = (
            pd.DataFrame(full_data, columns=['Arrival_Times', 'Arrival_Process', 
                                             'User', 'Group']))

        return full_data, df_full_data

    def _simulate_single_user(self, g: int, u_num: int) -> np.array:
        """
        Simulate the multivariate processes for a single user within group g. It will
        return a numpy array of two columns, a row consisting of an arrival time and the 
        process to which the arrival  time belongs. Parameters:
            - g: the group to which the user belongs.
            - u: the user number.
        """
        lam_ints = np.zeros((self.P, self.P)) # Stores lambda_m^i(r_{j-1}) in the notation of Lim et al. (2016)
        r = 0. # Initialise the jump time 
        a_ip = np.zeros((self.P+1, self.P)) # Matrix s.t. a_ip[i,p] = a^i_{pj} on iteration j
        t_jp = [[] for _ in range(self.P)] # List of lists to store event j for process p

        while r < self.T_max:
            for p in range(self.P):
                a_ip[0,p] = np.random.exponential(scale=1/self.lam[p, g], size=1)[0] # Numpy parameterises by scale

                for i in range(self.P):
                    u = np.random.uniform(size=1)[0]
                    if u < (1 - np.exp(-lam_ints[i, p] / self.theta[g, i, p])):
                        a_ip[i+1,p] = (
                            -np.log(1 + self.theta[g, i, p] * np.log(1 - u) / lam_ints[i, p]) / self.theta[g, i, p]
                        )
                    else:
                        a_ip[i+1, p] = np.inf # Index shift cos we count 0 -> P inc.

            # Find the minimum value of a_ip (call this d) and update r
            d = np.min(a_ip)
            r += d

            if r > self.T_max:
                pass
            else:
                min_idx = np.argwhere(a_ip == np.min(a_ip)).reshape((2,))
                i_star = min_idx[0]; p_star = min_idx[1]

                # Update lam_ints matrix
                for p in range(self.P):
                    for i in range(self.P):
                        if i == p_star:
                            lam_ints[i,p] *= np.exp(-self.theta[g, i, p] * d)
                            lam_ints[i,p] += self.beta[g, i, p]
                        else:
                            lam_ints[i,p] *= np.exp(-self.theta[g, i, p] * d)

                # Store the time in the appropriate list
                t_jp[p_star].append(r)
                
        # Concatenate the arrival times lists into a numpy array 
        times_arr = np.zeros((1,2))
        for i in range(len(t_jp)):
            group_id = np.array([i] * len(t_jp[i]))
            times_arr = np.vstack((times_arr, np.array([t_jp[i], group_id]).T))
        times_arr = times_arr[1:,:]
        times_arr = np.concatenate(
            [times_arr,
            np.array([u_num] * len(times_arr)).reshape((len(times_arr), 1))],
            axis=1
        )

        return times_arr

    def _compute_single_excitation(self, beta, theta, t_arrival, t_eval):
        """
        Compute the excitation function (without the constant term) for
        one of the processes in a multivariate process. This can then be
        summed to compute the total excitation function. Parameters:
            - beta: excitation parameter.
            - theta: decay parameter.
            - t_arrival: array of the arrival times.
            - t_eval: array of times to evaluate at. 
        """
        exc_func = np.array([(beta * np.exp(-theta * (t - t_arrival[t_arrival < t]))).sum() 
                    for t in t_eval])

        return exc_func

    def _compute_full_intensity_up(self, p: int, g: int, t_eval: np.array):
        """ 
        Compute the full conditional intensity function for a single user and process 
        combination. Namely, this will sum the contributions from each process and 
        add the basline intensity. Parameters:
            - p: process we consisder.
            - g: group the user belongs to.
            - t_eval: an array of the time values to evaluate the excitation
                    function at.
        """
        # Vector of betas for process p within group g
        betas_p = self.beta[g, :, p]
        # Vector of betas for process p within group g
        thetas_p = self.theta[g, :, p]
        # Lambda value for process p within group g
        lam_p = self.lam[p, g]

        # Initialise the excitation function with the baseline intensity.
        # We then loop over the processes and add to this list to compute total 
        # value at time t.
        ints_func = np.array([lam_p] * len(t_eval), dtype=float)
        for i in range(self.P):
            # List of arrival times to process i of P  
            t_arrival = self.t_arrivals_all[self.t_arrivals_all[:,1] == i, 0]
            ints_func += (
                self._compute_single_excitation(betas_p[i], thetas_p[i], t_arrival, t_eval)
            )

        return ints_func

    def simulate_and_plot(self, p: int, g: int, u: int, title='', save=False, file_name=None,):
        """
        Input an array containing evaluations of an intensity function and an
        array of times corresponding to where the intensity function was evaluated.
        This is then plotted. Parameters:
            - p: process we consider.
            - g: group the user belongs to.
            - u: user we consider within the group.
            - ints_func: array of intensity evaluations.
            - t_arrivals_all: array with column 1 being a list of arrival times and 
                              column 2 the process to which that arrival time belonged.
        """
        np.random.seed(1)
        if u >= self.G[g]:
            return ValueError(f"Group {g} has no user {u}")

        t_eval = np.linspace(0, self.T_max, self.T_max ** 3)

        # Simulate
        full_data, _ = self.simulate()

        # Store all arrivals for user u in group g
        self.t_arrivals_all = full_data[((full_data[:,2] == u) & (full_data[:,3] == g)), :]

        ints_func = self._compute_full_intensity_up(p, g, t_eval)
        baseline = ints_func.min()
        plt.plot(t_eval, ints_func);
        for i in range(self.P):
            t_arrival = self.t_arrivals_all[self.t_arrivals_all[:,1] == i, 0]
            plt.scatter(t_arrival, 
                        np.tile([-0.1 * (i + 1)], len(t_arrival)), # Jittered for visualisation
                        marker='X',  
                        linewidths=0.1);
        plt.title(title)
        plt.xlabel('t'); plt.ylabel(r"$\lambda(t)$")

        if save:
            if file_name == None:
                return ValueError('Please supply a file name.')
            else:
                plt.savefig(file_name + '.pdf')
        else:
            plt.show()
# #%%
# mu = np.array([[0.9, 0.4],
#                [1.1, 0.5]])

# beta = np.array(([[[2.0, 0.5],
#                    [0.3, 1.3]],

#                   [[4.0, 2.0],
#                    [2.0, 4.5]]]))

# theta = np.array(([[[2.2, 1.0],
#                     [1.0, 1.5]],

#                    [[7.5, 5.0],
#                     [5.0, 9.0]]]))

# ghe = GroupedExponentialHawkes(G=np.array([20, 20]), P=2, lam=mu, beta=beta,theta=theta)
# full_data, df_full_data = ghe.simulate()
# df_full_data_drop = df_full_data.drop(columns=['Group'])
# df_full_data_drop.to_csv('test_data.csv', index=False, header=False)
# #%%

