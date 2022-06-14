from typing import Dict, Any

import pandas as pd
from sdv.evaluation import evaluate


def run_sdv_analysis(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> Dict[str, Any]:

    """
    This statistic is taken from the following paper which outlines the approach in more detail:
    https://www.researchgate.net/publication/233708235_Evaluating_Goodness-of-Fit_Measures_for_Synthetic_Microdata

    inputs:
    - real data
    - synthetic data
    returns:
    - Dict containing the full evaluate breakdown and the mean score.
    """

    df = evaluate(
        real_data=real_data.fillna(0),
        synthetic_data=synthetic_data.fillna(0),
        aggregate=False,
    )

    summary_score = df.normalized_score.mean()
    return dict(sdv_metrics=df, sdv_summary_score=summary_score)
