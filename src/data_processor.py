"""
This modules contains classes and methods pertaining to data cleaning
and processing. Currently, they are limited in functionality and perform
only basic tests and alterations.
"""

import numpy as np
import pandas as pd

class DataProcessor:
    
    def __init__(self, df_raw, P, U, T_max) -> None:
        self.df_raw = df_raw; self.P = P; self.T_max = T_max
        self.U = U

    def processData(self) -> np.array:
        
        # Ensure types are correct
        self.df_raw = self.df_raw.astype({'Arrival_Times': 'float64',
                                          'Arrival_Process': 'int32',
                                          'User': 'int32'})
        
        # Drop Nans
        self.df_raw.dropna(inplace=True)

        # Check inputted number of users matches the number in the data
        if self.df_raw.User.nunique() != self.U:
            raise ValueError("""The number of users in the dataset does not match
                            the specified number of users.""")

        # Check that the number of processes match with P
        num_unique = self.df_raw.Arrival_Process.nunique()
        unique = np.array(self.df_raw.Arrival_Process.unique())
        unique.sort()
        
        if num_unique != self.P:
            raise ValueError("""The inputted number of processes does not match
                            the number of unique values for the process
                            enumeration in the data set.""")
        
        if unique.max() != (self.P-1):
            # If length is the same, but maximum is not the same, create
            # a mapping
            mapping_dict = {}
            keys = np.arange(self.P)
            for key, value in zip(unique, keys):
                mapping_dict[key] = value
            self.df_raw.Arrival_Process = self.df_raw.Arrival_Process.replace(mapping_dict)

        # Check that we have no negative values, and that all values fall within [0,T_max]
        if self.df_raw.Arrival_Times.min() <= 0:
            raise ValueError("Ensure all arrival_times are positive.")
        elif self.df_raw.Arrival_Times.max() > self.T_max:
            raise ValueError("Ensure all arrival_times do not exceed T_max")

        # There will be errors if a user has no arrivals to a process. This 
        # extracts which users are missing arrivals and the processes that
        # are missing
        result = (self.df_raw.groupby('User')['Arrival_Process']
                  .apply(lambda x: set(np.arange(self.P)).difference(x))
        )
        missing_values = result[result.apply(len) > 0]
        # This is a pd.series with index of the user and values of
        # lists containing the number of the process with no arrivals
        missing_elements = missing_values.apply(lambda x: list(x))
        # Create a data frame of the users and the processes with missing
        # arrivals. Add a column of None to this for Arrival_Times
        rows_to_add = (
            pd.DataFrame(
                {'Arrival_Times': 0,
                 'Arrival_Process': missing_elements.values, 
                 'User': missing_elements.index.values})
                 .explode('Arrival_Process', ignore_index=True))
        
        # Add these rows to df_raw
        self.df_raw = pd.concat([self.df_raw, rows_to_add], ignore_index=True)

        return self.df_raw
