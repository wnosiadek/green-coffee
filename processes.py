"""
Module for simplifying possible processing methods

Requires installation of 'pandas'

Functions
---------
simplify_processes
    Simplifies processing method names to integer ids
"""

import pandas
import data


def simplify_processes(train_features: pandas.Series | pandas.DataFrame,
                       train_processes: pandas.Series,
                       test_features: pandas.Series | pandas.DataFrame,
                       test_processes: pandas.Series,
                       print_map: bool = True) \
        -> tuple[pandas.Series | pandas.DataFrame, pandas.Series, 
                 pandas.Series | pandas.DataFrame, pandas.Series]:
    """
    Simplifies processing method names to integer ids

    Parameters
    ----------
    train_features: pandas.Series | pandas.DataFrame
        Chosen coffee characteristics for training
    train_processes: pandas.Series
        Processing methods for training
    test_features: pandas.Series | pandas.DataFrame
        Chosen coffee characteristics for testing
    test_processes: pandas.Series
        Processing methods for testing
    print_map: bool, default True
        Whether to print the {process: process_id} map

    Returns
    -------
    tuple[pandas.Series | pandas.DataFrame, pandas.Series, pandas.Series | pandas.DataFrame, pandas.Series]
        Input characteristics and processing methods simplified with the {process: process_id} map
    """

    # assign an integer id to each processing method
    processes_map = {process: process_id 
                     for process_id, process 
                     in enumerate([process
                                   for process, many in train_processes.value_counts().gt(2).items() if many])}
    # {'washed': 0, 'natural': 1, 'pulped natural': 2, 'anaerobic': 3}

    # transform given data using the mapping
    train_processes = train_processes.map(processes_map)
    test_processes = test_processes.map(processes_map)
    # drop any mistakes using 'data.py'
    train_processes, train_features = data.drop_missing(train_processes, train_features)
    test_processes, test_features = data.drop_missing(test_processes, test_features)

    # print the mapping for reference
    if print_map:
        print(f'\n{processes_map}')

    return train_features, train_processes, test_features, test_processes
