import pandas as pd
from pandas_profiling import ProfileReport


def run_pandas_profiling(real_data: pd.DataFrame, synthetic_data: pd.DataFrame):
    """
    This function runs the pandas profile for the real and synthetic datasets. More information on pandas profiling and
    outputs can be found here: https://pandas-profiling.ydata.ai/docs/master/index.html.

    inputs:
        - real data
        - synthetic data

    returns:
        - No return. Generates a html report which will appear in data/08_reporting/.
    """

    real_data_profile = ProfileReport(real_data, title="Pandas Profiling Report")
    real_data_profile.to_file("data/08_reporting/real_data_profile.html")

    synthetic_data_profile = ProfileReport(
        synthetic_data, title="Pandas Profiling Report"
    )
    synthetic_data_profile.to_file("data/08_reporting/synthetic_data_profile.html")
