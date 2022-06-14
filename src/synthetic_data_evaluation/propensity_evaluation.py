from typing import Dict

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from data_preparation.evaluation_data_prep import combine_data, save_image_as_bytes


def propensity_score_logestic_regression(
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
    show_plot: bool = True,
) -> Dict[str, float]:

    """
    This function checks whether a model can differentiate between the real and synthetic data using
    a logistic regression model trained on input data.

    inputs:
        - real data
        - synthetic data
        - show plot - will display plot when True selected.

    returns:
        - a dictionary containing the calculated mse value and the figure saved as "utf-8" string.
    """

    # combine dataset
    data = combine_data(real_data, synthetic_data)
    x = data[data.columns.drop("synthetic_data")]

    # logestic regression to predict probabilities
    clf = LogisticRegression(random_state=42, max_iter=300)
    clf.fit(x, data["synthetic_data"])
    ps = clf.predict_proba(x)[:, 1]
    data_ps = data.assign(propensity_score=ps)

    # plot
    if show_plot:
        fig, ax = plt.subplots(1, 2, figsize=(30, 10))
        syn = data_ps[data_ps["synthetic_data"] == 1]
        syn.hist("propensity_score", ax=ax[0], alpha=0.7)

        real = data_ps[data_ps["synthetic_data"] == 0]
        real.hist("propensity_score", color="r", ax=ax[0], alpha=0.3)

        data_ps.boxplot("propensity_score", by="synthetic_data", ax=ax[1])

    # Calculate MSE
    mse = (np.square(ps - 0.5)).mean()

    return dict(mse=mse, figure=save_image_as_bytes())
