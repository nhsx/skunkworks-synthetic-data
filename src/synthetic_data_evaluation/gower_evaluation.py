from typing import Dict

import pandas as pd
import gower


def gower_analysis(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> Dict[str, float]:
    """
    This function calculates the gower distance for each row in the
    synthetic dataset compared to each row in the real dataset.
    The function highlights the average closest distance and the distance
    most far away.
    This function calculates the gower distance using:
    https://pypi.org/project/gower/

    inputs:
        - real data
        - synthetic data

    returns:
        - a dictionary containing the minimum and maximum  gower dsitance.
    """

    # Calculate Gower Distance
    gower_distance = pd.DataFrame(
        gower.gower_matrix(data_x=synthetic_data, data_y=real_data)
    )

    # Calculate Gower Distance
    max_min = gower_distance.describe().transpose()[["min", "max"]]

    return dict(
        gower_distance_average_min=max_min["min"].mean(),
        gower_distance_average_max=max_min["max"].mean(),
    )
