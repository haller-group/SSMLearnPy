from pathlib import Path
import logging
from types import TracebackType

import pandas as pd
import numpy as np

logger = logging.getLogger("file_handler")

def get_vectors(dir: str)-> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    The function reads from a folder all the csv files and output the vectors
    time and trajectories for subsequent analysis using SSMLearn.

    The files should have the on the first column the time stamps and the other columns must contains
    the coordinates. E.g.:
    [
        time,   x_1,    x_2,    ...,    x_n
        time_1  x_1_1,  x_2_1,  ...,    x_n_1
        ...,    ...,    ...,    ...,    ...
        time_m, x_1_m,  x_2_m,  ...,    x_n_m
    ]
    Where the first row contains the headers. 
    Files should contains the same headers.
    """
    base_path = Path(dir)
    dfs = []

    for f in base_path.glob("*.csv"):
        logger.info(f"Reading file: {f}")
        dfs.append(
            pd.read_csv(f)
        )

    assert len(dfs) > 0, "No CSV files found in the directory."

    # Time vector is reshaped into (1, m) - see notation in docstring
    time = [df.iloc[:,0].values for df in dfs]

    # Trajectories vector is reshaped into (n, m) - see notation in docstring
    trajectories = [df.iloc[:,1:].values.T for df in dfs]

    return time, trajectories