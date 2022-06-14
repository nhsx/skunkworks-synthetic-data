import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from kedro.config import ConfigLoader

from data_preparation.evaluation_data_prep import save_image_as_bytes


def run_describe_feature_comparison(
    column: str,
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> pd.DataFrame:

    """
    This function calculates the summary statistics for a column which appears in
    both the real and synthetic data.

    inputs:
        - column name
        - real data
        - synthetic data

    returns:
        - dataframe containing summary statistics

    """

    # Create summary statistics table for synthetic and real datasets
    real_summary = pd.DataFrame(real_data[column].describe()).rename(
        columns={column: "real_data"}
    )
    syn_summary = pd.DataFrame(synthetic_data[column].describe()).rename(
        columns={column: "synthetic_data"}
    )

    # concat results together
    df = pd.concat([real_summary, syn_summary], axis=1)

    # take diffenrece if results are not categorical
    try:
        df["difference"] = df["real_data"] - df["synthetic_data"]
    except:
        pass

    return df


def run_categorical_feature_comparison(
    column: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> str:

    """
    This function runs a summary comparison of categories in the real and synthetic dataset. This includes:
        - counts per category in each dataset
        - proportions of each category in each dataset

    inputs:
        - column name
        - real data
        - synthetic data

    returns:
        - save_image_as_bytes() - the plot is returned as a "utf-8" string so it can be stored in the kedro
        pipline and appear in the html report.

    """

    # Define plots
    fig = plt.figure(figsize=(15, 15))
    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    axes = [ax1, ax2, ax3, ax4]

    # Plot categorical counts for real data
    real_data[column].value_counts().plot(kind="barh", ax=axes[0])
    axes[0].set_title("Real data counts per category", fontsize=15)
    axes[0].set_ylabel("Count of values", fontsize=10)

    # Plot categorical proportions for real data
    real_data[column].value_counts().plot(kind="pie", ax=axes[1], autopct="%1.1f%%")
    axes[1].set_title("Real Data category proportions", fontsize=15)

    # Plot categorical counts for synthetic data
    synthetic_data[column].value_counts().plot(kind="barh", ax=axes[2])
    axes[2].set_title("Synthetic Data", fontsize=15)
    axes[2].set_ylabel("Count of values", fontsize=10)

    # Plot categorical proportions for synthetic data
    synthetic_data[column].value_counts().plot(
        kind="pie", ax=axes[3], autopct="%1.1f%%"
    )
    axes[3].set_title("Synthetic data category proportions", fontsize=15)

    fig.suptitle(column, fontsize=20)

    # Return plot as "utf-8" string
    return save_image_as_bytes()


def run_numeric_feature_comparison(
    column: str, real_data: pd.DataFrame, synthetic_data: pd.DataFrame
):

    # Sprint subplots
    real_data["real_data"] = "real_data"
    synthetic_data["real_data"] = "synthetic_data"
    data = pd.concat([real_data, synthetic_data])

    fig = plt.figure(figsize=(25, 7))
    ax1 = plt.subplot(1, 3, 1)
    ax2 = plt.subplot(1, 3, 2)
    ax3 = plt.subplot(1, 3, 3)
    axes = [ax1, ax2, ax3]

    fig.suptitle(column, fontsize=20)

    # Distribution plot comparison
    real_data[column].plot.density(color="b", alpha=0.5, ax=axes[0], label="real_data")
    synthetic_data[column].plot.density(
        color="r", alpha=0.5, ax=axes[0], label="synthetic_data"
    )
    axes[0].legend()

    # Boxplot
    sns.boxplot(data=data, x="real_data", y=column, ax=axes[1])
    axes[1].legend()

    # Cumulative plot
    synthetic_data[column].cumsum().plot(ax=axes[2])
    real_data[column].cumsum().plot(ax=axes[2])
    axes[2].legend()
    return save_image_as_bytes()


def run_feature_count_nulls(
    column: str,
    real_data: pd.DataFrame,
    synthetic_data: pd.DataFrame,
) -> pd.DataFrame:

    real_missing = (real_data[column].isna().sum() / len(real_data[column])) * 100
    syn_missing = (
        synthetic_data[column].isna().sum() / len(synthetic_data[column])
    ) * 100
    df = pd.DataFrame(
        {
            "real data % missing": [real_missing],
            "synthetic data % missing": [syn_missing],
        },
    )
    return df


def run_feature_comparison(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, column: str
) -> None:
    """
    This function runs the feature comparison checks for the column specified in the
    dataset and completes the following checks:
    - Runs the description statistic for the column
    - Identifys nulls in the column
    - Runs categorical or numeric feature comparison as specified in the conf/base/parameters file
    """

    # Description statistics
    description = run_describe_feature_comparison(column, real_data, synthetic_data)
    # Missing values
    nulls = run_feature_count_nulls(column, real_data, synthetic_data)

    # Load categorical values as defined in conf/base/parameter
    conf_loader = ConfigLoader("conf/base")
    params = conf_loader.get("parameters*", "parameters*/**")

    if column in params["categorical_features"]:
        figure = run_categorical_feature_comparison(column, real_data, synthetic_data)
    else:
        figure = run_numeric_feature_comparison(column, real_data, synthetic_data)

    return dict(description=description, nulls=nulls, figure=figure)
