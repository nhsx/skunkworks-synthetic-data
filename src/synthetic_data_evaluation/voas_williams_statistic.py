import pandas as pd
import numpy as np


def voas_williams_statistic(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> float:
    """
    This statistic is taken from the following paper which outlines the approach in more detail:
    https://www.researchgate.net/publication/233708235_Evaluating_Goodness-of-Fit_Measures_for_Synthetic_Microdata

    inputs:
        - real data
        - synthetic data
        - show plot - will display plot when True selected.
    returns:
        - Prediction Mean Sqaure Error
    """

    # Calculate the numerator and denominator
    n = (synthetic_data - real_data) ** 2
    d = (synthetic_data + real_data) / 2

    # Calculate the fraction
    divide = n / d

    # Tidy DataFrame
    divide.fillna(0, inplace=True)
    divide.replace(np.inf, 0, inplace=True)

    # Sum the dataframe
    total = divide.values.sum()

    # Count the number of records
    real_data_records = len(real_data)
    synthetic_data_records = len(synthetic_data)
    N = real_data_records + synthetic_data_records

    # Calculate pMSE
    pMSE = total / (8 * N)

    return pMSE
