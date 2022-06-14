from functools import partial

from kedro.pipeline import Pipeline, node
from kedro.config import ConfigLoader

from synthetic_data_evaluation import pca_evaluation
from synthetic_data_evaluation.sdv_evaluation import run_sdv_analysis
from synthetic_data_evaluation.collision_analysis import collision_analysis
from synthetic_data_evaluation.propensity_evaluation import (
    propensity_score_logestic_regression,
)
from synthetic_data_evaluation.gower_evaluation import gower_analysis
from data_preparation.evaluation_data_prep import *
from synthetic_data_evaluation.feature_comparison_evaluation import (
    run_feature_comparison,
)
from synthetic_data_evaluation.voas_williams_statistic import voas_williams_statistic
from synthetic_data_evaluation.pandas_profiling_report import run_pandas_profiling


def eval_pipeline(real_data: str, synthetic_data: str, **kwargs) -> Pipeline:
    """
    This is the pipeline is to evaluate the real and synthetic datasets.

    The pipeline follows this process:
    - Selects the columns to be used in the analysis from the config
    - Converts time features to integer
    - Feature analysis - includes distributions and summary statisitics
    - Fill missing values with 0
    - SDV analysis
    - Covert categorical variables to binary
    - Voas Williams statistic
    - Gower statistic
    - Propensity score logestic regression
    - Collision analysis
    - PCA

    """

    nodes = []
    nodes.append(
        node(
            func=run_pandas_profiling,
            inputs=[real_data, synthetic_data],
            outputs=None,
            name="pandas_profiling",
        )
    )

    # select columns specified in conf/base/paramaters.yml - all features
    # --------------------------------------------------------------------
    nodes.append(
        node(
            func=select_columns,
            inputs=[
                real_data,
                synthetic_data,
                "params:all_features",
            ],
            outputs=dict(
                real_data_columns="real_data_columns",
                synthetic_data_columns="synthetic_data_columns",
            ),
            name="select_columns",
            tags="data_preparation",
        )
    )

    # convert time columns to int as specified in conf/base/paramaters.yml - time_features
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=convert_time_features,
            inputs=[
                "real_data_columns",
                "synthetic_data_columns",
                "params:time_features",
            ],
            outputs=dict(
                real_data="real_data_time_cleaned",
                synthetic_data="synthetic_data_time_cleaned",
            ),
            name="convert_time_features",
            tags="data_preparation",
        )
    )

    # feature analysis
    # --------------------------------------------------------------------------
    conf_loader = ConfigLoader("conf/base")
    params = conf_loader.get("parameters*", "parameters*/**")
    for column in params["all_features"]:
        nodes.append(
            node(
                func=partial(run_feature_comparison, column=column),
                inputs=["real_data_time_cleaned", "synthetic_data_time_cleaned"],
                outputs=dict(
                    description=f"{column}_description",
                    nulls=f"{column}_nulls",
                    figure=f"{column}_figure",
                ),
                tags=["data_evalulation_test", "images_as_base_64"],
                name=f"{column}",
            )
        )

    # filla missing values with 0
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=fill_missing_values_synthetic_real,
            inputs=["real_data_time_cleaned", "synthetic_data_time_cleaned"],
            outputs=dict(
                real_data="real_data_nulls_cleaned",
                synthetic_data="synthetic_data_nulls_cleaned",
            ),
            name="fill_missing_values_synthetic_real",
            tags="data_preparation",
        )
    )

    # SDV Analysis
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=run_sdv_analysis,
            inputs=["real_data_nulls_cleaned", "synthetic_data_nulls_cleaned"],
            outputs=dict(
                sdv_summary_score="sdv_summary_score", sdv_metrics="sdv_metrics"
            ),
            name="sdv_analysis",
            tags="data_evalulation_test",
        )
    )

    # convert categorical features to binary varaibles
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=convert_categorical_features,
            inputs=[
                "real_data_nulls_cleaned",
                "synthetic_data_nulls_cleaned",
                "params:categorical_features",
            ],
            outputs=dict(
                real_data="real_data_cleaned",
                synthetic_data="synthetic_data_cleaned",
            ),
            name="convert_categorical_features",
            tags="data_preparation",
        )
    )

    # voas williams stat
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=voas_williams_statistic,
            inputs=["real_data_cleaned", "synthetic_data_cleaned"],
            outputs="pMSE",
            name="voas_williams_statistic",
            tags="data_evalulation_test",
        )
    )

    # gower stats
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=gower_analysis,
            inputs=["real_data_cleaned", "synthetic_data_cleaned"],
            outputs=dict(
                gower_distance_average_min="average_min_gower_distance",
                gower_distance_average_max="average_max_gower_distance",
            ),
            name="gower_analysis",
            tags="data_evalulation_test",
        )
    )

    # propensity score logestic regression
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=propensity_score_logestic_regression,
            inputs=["real_data_cleaned", "synthetic_data_cleaned"],
            outputs=dict(
                mse="mse", figure="propensity_score_logestic_regression_figure"
            ),
            name="propensity_score_logestic_regression",
            tags=["data_evalulation_test", "images_as_base_64"],
        )
    )

    # collision analysis
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=collision_analysis,
            inputs=[
                "real_data_cleaned",
                "synthetic_data_cleaned",
            ],
            outputs=dict(number_of_rows="number_of_rows", same_row="same_row"),
            name="collision_analysis",
            tags="data_evalulation_test",
        )
    )

    # pca
    # --------------------------------------------------------------------------
    pca = pca_evaluation.pca_evaluation(n_components=2)
    nodes.append(
        node(
            func=pca.run_comparison,
            inputs=[
                "real_data_cleaned",
                "synthetic_data_cleaned",
            ],
            outputs="pca_evaluation_figure",
            name="pca_evaluation",
            tags=["data_evalulation_test", "images_as_base_64"],
        )
    )

    return Pipeline(nodes)

def support_eval_pipeline(real_data: str, synthetic_data: str, **kwargs) -> Pipeline:
    """
    This is the pipeline is to evaluate the real and synthetic datasets.

    The pipeline follows this process:
    - Selects the columns to be used in the analysis from the config
    - Converts time features to integer
    - Feature analysis - includes distributions and summary statisitics
    - Fill missing values with 0
    - SDV analysis
    - Covert categorical variables to binary
    - Voas Williams statistic
    - Gower statistic
    - Propensity score logestic regression
    - Collision analysis
    - PCA

    """

    nodes = []
    nodes.append(
        node(
            func=run_pandas_profiling,
            inputs=[real_data, synthetic_data],
            outputs=None,
            name="pandas_profiling",
        )
    )

    # select columns specified in conf/base/paramaters.yml - all features
    # --------------------------------------------------------------------
    nodes.append(
        node(
            func=select_columns,
            inputs=[
                real_data,
                synthetic_data,
                "params:all_support_features",
            ],
            outputs=dict(
                real_data_columns="real_data_columns",
                synthetic_data_columns="synthetic_data_columns",
            ),
            name="select_columns",
            tags="data_preparation",
        )
    )


    # feature analysis
    # --------------------------------------------------------------------------
    conf_loader = ConfigLoader("conf/base")
    params = conf_loader.get("parameters*", "parameters*/**")
    for column in params["all_support_features"]:
        nodes.append(
            node(
                func=partial(run_feature_comparison, column=column),
                inputs=["real_data_columns", "synthetic_data_columns"],
                outputs=dict(
                    description=f"{column}_description",
                    nulls=f"{column}_nulls",
                    figure=f"{column}_figure",
                ),
                tags=["data_evalulation_test", "images_as_base_64"],
                name=f"{column}",
            )
        )

    # filla missing values with 0
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=fill_missing_values_synthetic_real,
            inputs=["real_data_columns", "synthetic_data_columns"],
            outputs=dict(
                real_data="real_data_nulls_cleaned",
                synthetic_data="synthetic_data_nulls_cleaned",
            ),
            name="fill_missing_values_synthetic_real",
            tags="data_preparation",
        )
    )

    # SDV Analysis
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=run_sdv_analysis,
            inputs=["real_data_nulls_cleaned", "synthetic_data_nulls_cleaned"],
            outputs=dict(
                sdv_summary_score="sdv_summary_score", sdv_metrics="sdv_metrics"
            ),
            name="sdv_analysis",
            tags="data_evalulation_test",
        )
    )

    # convert categorical features to binary varaibles
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=convert_categorical_features,
            inputs=[
                "real_data_nulls_cleaned",
                "synthetic_data_nulls_cleaned",
                "params:support_categorical_features",
            ],
            outputs=dict(
                real_data="real_data_cleaned",
                synthetic_data="synthetic_data_cleaned",
            ),
            name="convert_categorical_features",
            tags="data_preparation",
        )
    )

    # voas williams stat
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=voas_williams_statistic,
            inputs=["real_data_cleaned", "synthetic_data_cleaned"],
            outputs="pMSE",
            name="voas_williams_statistic",
            tags="data_evalulation_test",
        )
    )

    # gower stats
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=gower_analysis,
            inputs=["real_data_cleaned", "synthetic_data_cleaned"],
            outputs=dict(
                gower_distance_average_min="average_min_gower_distance",
                gower_distance_average_max="average_max_gower_distance",
            ),
            name="gower_analysis",
            tags="data_evalulation_test",
        )
    )

    # propensity score logestic regression
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=propensity_score_logestic_regression,
            inputs=["real_data_cleaned", "synthetic_data_cleaned"],
            outputs=dict(
                mse="mse", figure="propensity_score_logestic_regression_figure"
            ),
            name="propensity_score_logestic_regression",
            tags=["data_evalulation_test", "images_as_base_64"],
        )
    )

    # collision analysis
    # --------------------------------------------------------------------------
    nodes.append(
        node(
            func=collision_analysis,
            inputs=[
                "real_data_cleaned",
                "synthetic_data_cleaned",
            ],
            outputs=dict(number_of_rows="number_of_rows", same_row="same_row"),
            name="collision_analysis",
            tags="data_evalulation_test",
        )
    )

    # pca
    # --------------------------------------------------------------------------
    pca = pca_evaluation.pca_evaluation(n_components=2)
    nodes.append(
        node(
            func=pca.run_comparison,
            inputs=[
                "real_data_cleaned",
                "synthetic_data_cleaned",
            ],
            outputs="pca_evaluation_figure",
            name="pca_evaluation",
            tags=["data_evalulation_test", "images_as_base_64"],
        )
    )

    return Pipeline(nodes)
