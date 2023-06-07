"""
Module for simplifying possible processing methods

Requires installation of 'pandas'

Functions
---------
simplify_processes
    Simplifies processing method names to integer ids
"""

import pandas


def simplify_processes(train_processes: pandas.Series,
                       test_processes: pandas.Series,
                       print_map: bool = True)\
        -> tuple[pandas.Series, pandas.Series]:
    """
    Simplifies processing method names to integer ids

    Parameters
    ----------
    train_processes: pandas.Series
        Processing methods for training
    test_processes: pandas.Series
        Processing methods for testing
    print_map: bool, optional
        Whether to print the {process: process_id} map (default is True)

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        Input processing methods data simplified with the {process: process_id} map
    """

    # assign an integer id to each processing method
    processes_map = {process: process_id for process_id, process in enumerate(train_processes.unique())}
    # {'pulped natural': 0, 'natural': 1, 'double fermente': 2, 'washed': 3, 'red honey': 4, 'semi-washed': 5,
    #  'prolonged ferme': 6, 'wet hulled': 7, 'anaerobic': 8, 'honey': 9}

    # transform given data using the mapping
    train_processes = train_processes.map(processes_map)
    test_processes = test_processes.map(processes_map)

    # print the mapping for reference
    if print_map:
        print(processes_map)

    return train_processes, test_processes
