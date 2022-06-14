from typing import Dict

import pandas as pd


def collision_analysis(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> Dict[str, float]:
    """
    Collision analysis looks to see if any lines in the synthetic data match the
    lines in the real data.

    inputs:
    - real_data
    - synthetic_data

    returns:
    - a dictionary containing the dataframe of rows which appear in both dataframes and the number of these rows.
    """

    # Inner merge on all columns in the real and synthetic dataframes to find matching rows
    cols = real_data.columns[1:].tolist()
    df = real_data.merge(how="inner", on=cols, right=synthetic_data)
    return dict(same_row=df, number_of_rows=len(df))
