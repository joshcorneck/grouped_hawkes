"""
This is the entry point for the code. We run the function main, which 
will load, process and compute the EM parameter values for the data. 
"""
from src.data_processor import DataProcessor
from src.EM_update_reparam import EM

import argparse
import pandas as pd
import numpy as np


def main(data_file_path: str, N_EM_its: int, N_grad_steps: int, P: int, 
         U: int, H: int, T_max: float):
    """
    Parameters:
        - df_full_data: the dataset. This should take the form of having three 
            columns corresponding to: arrival times (float), process the arrival is 
            from (int), user number (int). Processes should be enumerated from 0 to P-1, 
            inline with Python enumeration, and analogously for users and groups. 
        - N_EM_its: number of iterations to run.
        - N_grad_steps: number of gradient ascent steps to run on each 
            iteration.
        - P: number of processes.
        - U: number of users.
        - H: number of groups.
        - T_max: upper limit of the observation window.
    Output:
        - psi: the parameter values. This is a np.array of shape (H, 1 + P + 2 * P ** 2).
               It is outputted as psi[h,0] = pi_h, psi[h,1:(P + 1)] = mu_h,
               psi[h, (P + 1):(P + 1 + P ** 2)] = beta_h and
               psi[h, (P + 1 + P ** 2):] = theta_h. See documentation for more
               detail.
    """
    # Load in the data csv file
    try:
        pd.read_csv(data_file_path)        
    except FileNotFoundError:
        print(f"Error: The file '{data_file_path}' was not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{data_file_path}' is empty or not a valid CSV file.")

    df_raw_data = pd.read_csv(data_file_path)
    df_raw_data.columns = ['Arrival_Times', 'Arrival_Process', 'User']

    # Process the data
    dp = DataProcessor(df_raw_data, P, U, T_max)
    df_processed = dp.processData()

    # Run the EM algorithm using the given parameters
    EM_inst = EM(df_processed, U, P, H, T_max)
    psi, gammas, _, _ = (
        EM_inst.EM_full(n_its=N_EM_its, n_grad_steps=N_grad_steps)
    )

    return psi, gammas

if __name__ == '__main__':

    # Parse arguments
    parser = argparse.ArgumentParser(description='Input global arguments.')
    parser.add_argument("data_file_path", type=str, help="path to data in csv form.")
    parser.add_argument("N_EM_its", type=int, help="number of iterations for EM.")
    parser.add_argument("N_grad_steps",  type=int, help="number of gradient ascent steps.")
    parser.add_argument("P", type=int, help="number of processes.")
    parser.add_argument("U", type=int, help="number of users.")
    parser.add_argument("H", type=int, help="number of groups.")
    parser.add_argument("T_max", type=int, help="upper limit of observation window.")

    args = parser.parse_args()

    # Run the EM algorithm
    psi, gammas = main(args.data_file_path, args.N_EM_its, args.N_grad_steps, args.P, args.U, 
         args.H, args.T_max)
    
    # ADD METHOD FOR SAVING DOWN psi AND gammas
