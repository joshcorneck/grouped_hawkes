"""
This module contains classes and functions for running the 
EM-update for an inputted data set. The first class (EM) is
the class for implementing the method, and the remaining 
functions are helper functions for the implementation.
"""
from typing import List
import numpy as np
import pandas as pd
from scipy.special import logsumexp

#####################################################################
################## MAIN CLASS FOR THE EM ALGORITHM ##################
#####################################################################

class EM:
    """
    
    """
    def __init__(self, df_full_data: pd.DataFrame, U: int, P: int, 
                 H: int, T: float) -> None:
        """
        Parameters:
            - df_full_data: data set to run the EM procedure on. 
            - U, P, H: the number of users, processes and groups.
            - T: the upper limit of the time interval. Assumed to 
                 be run on [0,T].
        """
        self.df_full_data = df_full_data
        self.U = U; self.P = P; self.H = H; self.T = T

    def _prep_data(self):
        """
        Prepare the data for EM procedure.
        """
        # Group by User and Arrival_Process and concatenate times into
        # a list
        df_grouped = (
            self.df_full_data.
            groupby(['User', 'Arrival_Process']).
            agg({'Arrival_Times': lambda x: x.tolist()}).
            reset_index()
        )

        # Establish a pandas array of lists for all arrival times for each user
        self.df_t_all = (
            df_grouped.
            groupby(['User']).
            agg({'Arrival_Times': lambda x: x.tolist()}).
            reset_index()
        )

    def _compute_ll(self, psi_tilde_m: np.array) -> np.array:
        """
        Computes the loglikelihood for each user over different assumed
        group memeberships for the current parameter values.
        Parameters:
            - psi_tilde_m: am array containing all parameters at the m-th
                           step of the update. These are the transformed
                           parameters.
        Output:
            - ell_m: the log-likelihood on step m.
        """
        # Initialise array of zeros
        ell_m = np.zeros((self.U, self.H))

        for u in range(self.U):
            t_u_all = (
                self.df_t_all[self.df_t_all.User == u].
                Arrival_Times.
                values[0]
            )
            for h in range(self.H):
                ell_m[u,h] = (
                    log_likelihood_u_h(psi_tilde_m[h,1:], self.T, self.P, t_u_all)
                )

        return ell_m
    
    def _param_transform(self, psi_m: np.array) -> np.array:
        """
        Computes the reparameterisation.
        Parameters:
            - psi_m: the untransformed parameters.
        Output:
            - psi_m_tilde: the transformed parameters.
        """
        psi_m_tilde_temp = np.zeros((self.H, 1 + self.P + 2 * self.P ** 2))
        psi_m_tilde_temp[:,0] = psi_m[:,0]
        psi_m_tilde_temp[:,1:(self.P+1)] = np.log(psi_m[:,1:(self.P+1)])
        psi_m_tilde_temp[:,(self.P + 1):(self.P + self.P**2 + 1)] = (
            np.log(psi_m[:,(self.P + 1):(self.P + self.P**2 + 1)])
        )
        psi_m_tilde_temp[:,(self.P + self.P**2 + 1):] = (
            np.log(
                psi_m[:,(self.P + self.P**2 + 1):] 
                - 
                psi_m[:,(self.P + 1):(self.P + self.P**2 + 1)]
            )
        )

        return psi_m_tilde_temp
    
    def _param_inv_transform(self, psi_m_tilde: np.array) -> np.array:
        """
        Inverts the parameter transform.
        Parameters:
            - psi_m_tilde: the transformed parameters.
        Output:
            - psi_m: the transformed parameters.
        """
        psi_m_temp = np.zeros((self.H, 1 + self.P + 2 * self.P ** 2, ))
        psi_m_temp[:, 0] = psi_m_tilde[:,0]
        psi_m_temp[:, 1:(self.P+1)] = np.exp(psi_m_tilde[:,1:(self.P+1)])
        psi_m_temp[:, (self.P + 1):(self.P + self.P**2 + 1)] = (
            np.exp(psi_m_tilde[:,(self.P + 1):(self.P + self.P**2 + 1)])
        )
        psi_m_temp[:, (self.P + self.P**2 + 1):] = (
            np.exp(psi_m_tilde[:,(self.P + 1):(self.P + self.P**2 + 1)])
            +
            np.exp(psi_m_tilde[:,(self.P + self.P**2 + 1):]) 
        )

        return psi_m_temp

    def _E_step(self, psi_tilde_m: np.array) -> np.array:
        """
        A function to run the expectation step of the algorithm to compute
        gamma_ik for iteration m. 
        Parameters:
            - psi_tilde_m: the (transformed) parameters on iteration m.
        Output:
            - gammas: a np.array of shape (U,H) with entry (u,h) being gamma_uh.
        """
        # Compute the loglikelihoods
        ell_m = self._compute_ll(psi_tilde_m)

        # Array to store the log-scale gammas
        log_gammas = np.zeros((self.U, self.H))

        # Iterate over the users
        for u in range(self.U):
            # Iterate over the groups
            for h in range(self.H):
                # Check if psi_m[h,0] is sufficiently small
                if psi_tilde_m[h,0] < 10e-4:
                    log_gammas[u,h] = -np.inf
                else:
                    log_gammas[u,h] = ell_m[u,h] + np.log(psi_tilde_m[h,0])
        
        # Array to store the normalised gammas
        gammas = np.zeros((self.U, self.H))

        # Normalise over the columns (using log-sum-exp trick)
        for u in range(self.U):
            gammas[u,:] = np.exp(log_gammas[u,:] - logsumexp(log_gammas[u,:]))

        return gammas

    def _M_step(self, gammas: np.array, psi_m_tilde: np.array, 
                n_grad_steps: int, a: float, b1: float, b2: float, 
                eps: float) -> np.array:
        """
        Runs the ADAM algorithm for n_grad_steps. Gradients are computed using the
        exponential parametrisation.
        Parameters:
            - gammas: the node-level group-membership probabiliites.
            - psi_m_tilde: the current (transformed) parameter values.
            - n_grad_steps: number of steps to take on each run.
            - a, b1, b2, eps: parameters of the ADAM algorithm.
        Output:
            - psi_m_tilde: the (transformed) parameters post ascent steps.
        """
        # Compute pi^(m) - this is simply the row mean of gammas
        # psi_m[:,0] = gammas.mean(axis=0)
        psi_m_tilde[:,0] = gammas.mean(axis=0) 

        for h in range(self.H):
            
            # Initialise first and second moment vectors as 0
            m = np.zeros((self.P + 2 * self.P ** 2, )) 
            v = np.zeros((self.P + 2 * self.P ** 2, ))

            # Initialise timestep and gradient for group h
            grad_count = 0; g_h = np.zeros((self.P + 2 * self.P ** 2, ))

            # Iterate for required number of time steps
            while grad_count < n_grad_steps:
                # Increment time step
                grad_count += 1
                for u in range(self.U):
                    # List of lists for all arrival times for user u
                    t_u_all = (
                                self.df_t_all[self.df_t_all.User == u].
                                Arrival_Times.
                                values[0]
                            )
                    g_h += gammas[u,h] * (-pdv_ll_u_full_reparam(psi_m_tilde[h,1:], self.T, 
                                                                 self.P, t_u_all))

                # Adam step
                m = b1 * m + (1 - b1) * g_h
                v = b2 * v + (1 - b2) * g_h ** 2
                m_hat = m/(1 - b1 ** grad_count)
                v_hat = v/(1 - b2 ** grad_count)
                psi_m_tilde[h,1:] = psi_m_tilde[h,1:] - a * m_hat / (np.sqrt(v_hat) + eps)
            
        return psi_m_tilde
    
    def EM_full(self, n_its: int, n_grad_steps: int, a=0.001, b1=0.9, 
                b2=0.999, eps=10e-7) -> np.array:
        """
        Runs the entire EM-update. 
        Parameters:
            - n_its: number of iterations to run of the EM steps.
            - n_grad_steps: number of steps to taken in the M step.
            - a, b1, b2, eps: parameters of ADAM algorithm.
        Output:
            - psi_m: the final parameter values.
            - gammas: the final group-level probabilities. Rows are
                      the nodes as enumerated in the data and columns
                      are groups, so (i,j) is probability of node i
                      being in group j. 
             
        """
        # Prepare the data
        self._prep_data()

        # Full parameter array for iteration m - 1 col for pi,
        # P cols for mu, P**2 cols for each of beta and theta
        psi_m_tilde = np.zeros((self.H, 1 + self.P + 2 * self.P ** 2))
        psi_m_tilde[:,0] = np.array([1/self.H] * self.H)
        
        mu_0 = (
            np.random.uniform(0.05,0.3,self.P * self.H).
            reshape((self.H, self.P), order='F')
        )
        beta_0 = (
            np.random.uniform(3.0,4.5,self.P ** 2 * self.H).
            reshape((self.H, self.P ** 2), order='F')
        )
        theta_0 = (
            np.random.uniform(4.5,6.0,self.P ** 2 * self.H).
            reshape((self.H, self.P ** 2), order='F')
        )

        psi_m_tilde[:,1:] = np.concatenate((mu_0, beta_0, theta_0), axis=1)
        psi_m_tilde = self._param_transform(psi_m_tilde)

        # Store iteration values
        gammas_store = np.zeros((n_its, self.U, self.H))
        psi_store = np.zeros((n_its, self.H, 1 + self.P + 2 * self.P ** 2))    

        # Run E and M steps until convergence (FOR NOW, FIXED NUMBER OF STEPS)
        # Check if group membership probabilities aren't changing
        its = 0
        while its < n_its:
            its += 1
            print(f"Iteration: {its} of {n_its}.")            
            gammas = self._E_step(psi_m_tilde)
            psi_m_tilde = self._M_step(gammas, psi_m_tilde, n_grad_steps, 
                                              a, b1, b2, eps)
            # Store for tracking convergence
            gammas_store[its-1,:,:] = gammas
            psi_store[its-1,:,:] = psi_m_tilde

        psi_m = self._param_inv_transform(psi_m_tilde)

        return psi_m, gammas, psi_store, gammas_store


#####################################################################
########## GENERAL FUNCTIONS FOR COMPUTING THE E & M-STEPS ##########
#####################################################################

# FUNCTION FOR THE A & \tilde{A} RECURSIONS
def AB_recursion(A_bool: bool, prev: float, t_upk: float, t_upk_minus: float, 
                 t_uj_all: np.array, beta_tilde_jph: float, theta_tilde_jph: float):
    """
    General function to compute A or B recursively. Parameters:
        - A_bool: Boolean to indicate if A or B. If A then True.
        - t_upk: the input at this time step. 
        - t_upk_minus: the input at the previous time step.
        - t_uj_all: all arrival times to process j for user u.
        - theta_tilde_jph: analogous for the decay parameters.
    """
    if t_upk == 0:
        return 0
    else: 
        # Find the times that contribute to the added term
        t_uj_take = t_uj_all[(t_upk_minus <= t_uj_all) & (t_uj_all < t_upk)]

        # If t_uj_take contains only 0's then set it to empty.
        if (t_uj_take == 0).all():
            t_uj_take = np.array([])

        # Compute term to scale prev.
        scale_factor = np.exp(-(np.exp(beta_tilde_jph) + np.exp(theta_tilde_jph)) * 
                              (t_upk - t_upk_minus))

        # Compute the exp and sum
        if len(t_uj_take != 0):
            if A_bool:
                return (scale_factor * prev + 
                        np.exp(-(np.exp(beta_tilde_jph) + np.exp(theta_tilde_jph)) * 
                               (t_upk - t_uj_take)).sum())
            else:
                return (scale_factor * prev + 
                        (t_uj_take * np.exp(-(np.exp(beta_tilde_jph) + np.exp(theta_tilde_jph)) * 
                                            (t_upk - t_uj_take))).sum())
        else:
            return (scale_factor * prev)

# CHECK
@np.vectorize
def vectorised_exponential(t_ujl, x, T):
    return np.exp(- x * (T - t_ujl))
    
#####################################################################
############ FUNCTIONS FOR EVALUATING THE LOG-LIKELIHOOD ############
#####################################################################

### FUNCTIONS TO COMPUTE THE LOG-LIKELIHOOD FOR USER U

# EVALUATION OF lambda_up (USER u, PROCESS p) FOR A GIVEN GROUP h AT TIME t_{upk}

def lambda_tupk_h(A: np.array, t_upk: float, t_prev: float, t_u_all: List[List], 
                  P: int, mu_tilde_ph: float, betas_tilde_ph: np.array, 
                  thetas_tilde_ph: np.array):
    """
    Function to evaluate lambda_up at t_upk for assumed group identity of h. 
    Parameters:
        - A: the values of A at t_{up(k-1)}. 
        - t_upk: the time to evaluate at.
        - t_prev: the previous evaluation time.
        - t_u_all: list of lists of all arrival times for user u (ordered by
                    the enumeration of the processes).
        - P: number of processes.
        - mu_tilde_ph: (transformed) background intensity for process p in group h.
        - betas_tilde_ph: (transformed) all excitation parameters for group h, 
                 organised such that entry j is the parameter for process j exciting 
                 process p. Shape (P,).s
        - thetas_tilde_ph: (transformed) all decay parameters for group h, 
                  organised such that entry j is the parameter for process j exciting 
                  process p. Shape (P,).
    """
    if not isinstance(betas_tilde_ph, np.ndarray):
        raise ValueError("Ensure betas_ph is of type numpy.ndarray.")
    elif not isinstance(thetas_tilde_ph, np.ndarray):
        raise ValueError("Ensure thetas_ph is of type numpy.ndarray.")

    for j in range(P):
        # Update A using the recursion, taking the current value as the previous.
        A[j] = AB_recursion(True, A[j], t_upk, t_prev, np.array(t_u_all[j]), 
                            betas_tilde_ph[j], thetas_tilde_ph[j])
    return A, np.exp(mu_tilde_ph) + (np.exp(betas_tilde_ph) * A).sum()

# EVALUATE THE LOG-LIKELIHOOD FOR USER u, PROCESS p GIVEN GROUP h

def log_likelihood_up_h(T: float, P: int, t_up_all: list, t_u_all: List[List], 
                        mu_tilde_ph: float, betas_tilde_ph: np.array, 
                        thetas_tilde_ph: np.array):
    """
    Function to evaluate ell_up given group h.
    Parameters:
        - T: upper bound of simulation window.
        - P: number of processes.
        - t_up_all: all arrival times for user u to process p.
        - t_u_all: all arrival times for user u (ordered by the enumeration of the processes).
        - mu_tilde_ph: (transformed) background intensity for process p for group h.
        - betas_tilde_ph: (transformed) containing all excitation parameters for group h, 
                 organised such that entry j is the parameter for process j exciting 
                 process p. Shape (P,).
        - thetas_tilde_ph: (transformed) containing all decay parameters for group h, 
                  organised such that entry j is the parameter for process j exciting 
                  process p. Shape (P,).
    """
    ###~~
    # Evaluate the sum of the log intensity
    ###~~

    # Empty arrays for the recursion. A is a place holder that gets updated for the
    # recusion on each run. B is an array of the intensity evaluations at each time 
    # point.
    A = np.zeros((P, )); B = np.zeros((len(t_up_all), ))

    # Initialise previous value as 0.
    t_prev = 0
    # If we have no arrival times to process p, set to 0 by default.
    if (np.array(t_up_all) == 0).all():
        sum_log_intensity = 0
    else:
        for k, t_upk in enumerate(t_up_all):
            A, B[k] = lambda_tupk_h(A, t_upk, t_prev, t_u_all, P, mu_tilde_ph, 
                                    betas_tilde_ph, thetas_tilde_ph)
            t_prev = t_upk
        sum_log_intensity = np.log(B).sum()

    ###~~
    # Evaluate the integral term
    ###~~
    integral_term = T * np.exp(mu_tilde_ph)
    for j in range(P):
        # Checks if no arrival times to process j for user u, in which case contribution
        # to the integral term is 0.
        if not (np.array(t_u_all[j]) == 0).all():
            integral_term += (
                (np.exp(betas_tilde_ph[j])/(np.exp(betas_tilde_ph[j]) + np.exp(thetas_tilde_ph[j]))) * 
                (1 - vectorised_exponential(np.array(t_u_all[j]), 
                                            np.exp(betas_tilde_ph[j]) + np.exp(thetas_tilde_ph[j]), 
                                            T)).sum()
            )

    return (sum_log_intensity - integral_term)

# FULL LIKELIHOOD EVALUATION FOR USER u ASSUMING GROUP h

def log_likelihood_u_h(params_tilde, T, P, t_u_all):
    """
    Function to compute the full loglikelihood for a user u assuming they
    belong to group h. Need to input parameters as one flattened numpy.ndarray, 
    ordered as (mu, beta, theta).
    """
    # Extract parameters
    mus_tilde_h = params_tilde[0:P]
    betas_tilde_h = params_tilde[P:(P + P**2)].reshape((P,P), order='F') 
    thetas_tilde_h = params_tilde[(P + P**2):].reshape((P,P), order='F')

    # Initialise loglikelihood value
    ll = 0

    # Iterate over all the groups
    for p in range(P):
        # Take the parameters and arrival times for process p
        mu_tilde_ph = mus_tilde_h[p]; betas_tilde_ph = betas_tilde_h[:,p] 
        thetas_tilde_ph = thetas_tilde_h[:,p]
        t_up_all = t_u_all[p]
        # Evaluate the loglikelihood for process p
        ll += log_likelihood_up_h(T, P, t_up_all, t_u_all, mu_tilde_ph, betas_tilde_ph, 
                                  thetas_tilde_ph)

    return ll

#####################################################################
############## FUNCTIONS FOR COMPUTING THE DERIVATIVES ##############
#####################################################################

### FUNCTION TO COMPUTE THE PARTIAL DERIVATIVES FOR USER u WITH RESPECT TO THE 
### PARAMETERS FOR PROCESS p_prime EXCITED BY j_prime

def pdv_ll_u_pprime_jprime_reparam(T, P, t_up_all, t_u_all, mu_tilde_h, 
                                   betas_tilde_h, thetas_tilde_h, p_prime, 
                                   j_prime):
    """
    This will take as input mu_tilde, beta_tilde, theta_tilde and return the derivatives 
    with respect to mu_tilde, beta_tilde, theta_tilde.
    """
    # Take the parameters for process p_prime
    mu_tilde_ph = mu_tilde_h[p_prime]; betas_tilde_ph = betas_tilde_h[:,p_prime]; 
    thetas_tilde_ph = thetas_tilde_h[:,p_prime]

    # Empty arrays for the recursion.
    A = np.zeros((P,)); B = np.zeros((len(t_up_all),))
    B_mu_tilde = B.copy(); B_beta_tilde = B.copy(); B_theta_tilde = B.copy()
    
    # Initialise previous value as 0.
    t_prev = 0; A_tilde = 0
    # If no arrivals to process p, keep as 0
    if not (np.array(t_up_all) == 0).all():
        for k, t_upk in enumerate(t_up_all):
            A_tilde = AB_recursion(False, A_tilde, t_upk, t_prev, np.array(t_u_all[j_prime]), 
                                   betas_tilde_ph[j_prime], thetas_tilde_ph[j_prime])
            A, B[k] = lambda_tupk_h(A, t_upk, t_prev, t_u_all, P, mu_tilde_ph, betas_tilde_ph, 
                                    thetas_tilde_ph)
            B_mu_tilde[k] = 1 / B[k]
            B_beta_tilde[k] = A[j_prime] / B[k]
            B_theta_tilde[k] = np.exp(betas_tilde_ph[j_prime]) * (A_tilde - t_upk * A[j_prime]) / B[k]
            t_prev = t_upk

    if not (np.array(t_u_all[j_prime]) == 0).all():
        integral_term = (
                (1 / (np.exp(betas_tilde_ph[j_prime]) + np.exp(thetas_tilde_ph[j_prime]))) * 
                (1 - vectorised_exponential(np.array(t_u_all[j_prime]), 
                                            np.exp(betas_tilde_ph[j_prime]) + np.exp(thetas_tilde_ph[j_prime]), 
                                            T)).sum()
            )
    
        first_sub_term = (
                (np.exp(betas_tilde_ph[j_prime]) / 
                (np.exp(betas_tilde_ph[j_prime]) + np.exp(thetas_tilde_ph[j_prime])) ** 2) * 
                (1 - vectorised_exponential(np.array(t_u_all[j_prime]), 
                                            np.exp(betas_tilde_ph[j_prime]) + np.exp(thetas_tilde_ph[j_prime]), 
                                            T)).sum()
            )

        second_sub_term = (
            (np.exp(betas_tilde_ph[j_prime]) / 
            (np.exp(betas_tilde_ph[j_prime]) + np.exp(thetas_tilde_ph[j_prime]))) *
            ((T - np.array(t_u_all[j_prime])) * 
            vectorised_exponential(np.array(t_u_all[j_prime]), 
                                   np.exp(betas_tilde_ph[j_prime]) + np.exp(thetas_tilde_ph[j_prime]), 
                                   T)).sum()
            )
    else:
        integral_term = 0; first_sub_term = 0; second_sub_term = 0

    pdv_mu_tilde = (B_mu_tilde.sum() - T)
    pdv_beta_tilde = (B_beta_tilde.sum() - integral_term)
    pdv_theta_tilde = (B_theta_tilde.sum() + first_sub_term - second_sub_term)

    # Multiply to transform derivatives for reparametrisation.
    return np.exp(mu_tilde_ph) * pdv_mu_tilde,\
           np.exp(betas_tilde_ph[j_prime]) * (pdv_beta_tilde + pdv_theta_tilde),\
           np.exp(thetas_tilde_ph[j_prime]) * pdv_theta_tilde
        
### FUNCTION TO COMPUTE ALL PARTIAL DERIVATIVES FOR USER u 

def pdv_ll_u_full_reparam(params_tilde, T, P, t_u_all):
    """
    """
    # Extract parameters
    mus_tilde_h = params_tilde[0:P]
    betas_tilde_h = params_tilde[P:(P + P**2)].reshape((P,P), order='F') 
    thetas_tilde_h = params_tilde[(P + P**2):].reshape((P,P), order='F')

    # Arrays to store derivatives
    pdv_mu_tilde = np.zeros((P,)); pdv_beta_tilde = np.zeros((P,P)); 
    pdv_theta_tilde = np.zeros((P,P))

    for p_prime in range(P):
        # Extract relevant arrival times
        t_up_all = t_u_all[p_prime]
        for j_prime in range(P):
            pdvs_tilde = pdv_ll_u_pprime_jprime_reparam(T, P, t_up_all, t_u_all, 
                                                        mus_tilde_h, betas_tilde_h, 
                                                        thetas_tilde_h, p_prime, j_prime)
            pdv_mu_tilde[p_prime] = pdvs_tilde[0]
            pdv_beta_tilde[j_prime, p_prime] = pdvs_tilde[1]
            pdv_theta_tilde[j_prime, p_prime] = pdvs_tilde[2]

    # Fortran for column-major
    pdvs_tilde_full = np.concatenate((pdv_mu_tilde, 
                                      pdv_beta_tilde.flatten('F'),
                                      pdv_theta_tilde.flatten('F')))

    return pdvs_tilde_full