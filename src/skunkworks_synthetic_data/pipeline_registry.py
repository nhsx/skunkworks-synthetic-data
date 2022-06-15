"""Project pipelines."""
from typing import Dict

from kedro.pipeline import Pipeline

import synthetic_data_evaluation.pipelines.data_evaluation_pipeline as data_evaluation
import synthetic_data_generation.pipelines.data_generation_pipeline as data_generation
import synthetic_data_generation.pipelines.demonstrator_pipeline as support_demo


def register_pipelines() -> Dict[str, Pipeline]:
    """Register the project's pipelines.

    Returns:
        A mapping from a pipeline name to a ``Pipeline`` object.
    """

    # Below Small, Medium and Large are used to name pipelines
    # Small refers to the smallest input file (table_one_11040)
    # Medium refers to the middle-sized input file (table_one_imbalanced_81795)
    # Large refers to the largest input file (table_one_imbalanced_217010)
    # The prefix of small, medium and large refers to the input file that is
    # generated and/or used to train the model for synthetic data generation

    load_data_pipeline = data_generation.load_data_pipeline()
    small_preproc_pipeline = data_generation.small_preproc_pipeline()
    medium_preproc_pipeline = data_generation.medium_preproc_pipeline()
    large_preproc_pipeline = data_generation.large_preproc_pipeline()

    small_synthetic_generation_pipeline = (
        data_generation.generate_synthetic_input_pipeline("table_one_11040")
    )
    medium_synthetic_generation_pipeline = (
        data_generation.generate_synthetic_input_pipeline("table_one_imbalanced_81795")
    )
    large_synthetic_generation_pipeline = (
        data_generation.generate_synthetic_input_pipeline("table_one_imbalanced_217010")
    )

    # Evalulation pipelines
    small_data_evaluation_pipeline = data_evaluation.eval_pipeline(
        real_data="table_one_11040", synthetic_data="synthetic_data_input"
    )
    med_data_evaluation_pipeline = data_evaluation.eval_pipeline(
        real_data="table_one_imbalanced_81795",
        synthetic_data="synthetic_data_input",
    )
    large_data_evaluation_pipeline = data_evaluation.eval_pipeline(
        real_data="table_one_imbalanced_217010",
        synthetic_data="synthetic_data_input",
    )
    support_data_evaluation_pipeline = data_evaluation.support_eval_pipeline(
        real_data="support_input_data", synthetic_data="support_synthetic_data"
    )

    custom_data_evaluation_pipeline = data_evaluation.eval_pipeline(
        real_data="custom_real_data", synthetic_data="custom_synthetic_data"
    )

    support_generate_pipeline = support_demo.support_pipeline()

    support_demo_pipeline = Pipeline(
        [support_generate_pipeline, support_data_evaluation_pipeline]
    )
 
    small_end_to_end = Pipeline(
        [
            load_data_pipeline,
            small_preproc_pipeline,
            small_synthetic_generation_pipeline,
            small_data_evaluation_pipeline,
        ]
    )
    medium_end_to_end = Pipeline(
        [
            load_data_pipeline,
            medium_preproc_pipeline,
            medium_synthetic_generation_pipeline,
            med_data_evaluation_pipeline,
        ]
    )
    large_end_to_end = Pipeline(
        [
            load_data_pipeline,
            large_preproc_pipeline,
            large_synthetic_generation_pipeline,
            large_data_evaluation_pipeline,
        ]
    )

    return {
        "__default__": small_end_to_end,
        "small_data_evaluation_pipeline": small_data_evaluation_pipeline,
        "med_data_evaluation_pipeline": med_data_evaluation_pipeline,
        "large_data_evaluation_pipeline": large_data_evaluation_pipeline,
        "custom_data_evaluation_pipeline": custom_data_evaluation_pipeline,
        "support_data_evaluation_pipeline": support_data_evaluation_pipeline,
        "small_preproc_pipeline": small_preproc_pipeline,
        "medium_preproc_pipeline": medium_preproc_pipeline,
        "large_preproc_pipeline": large_preproc_pipeline,
        "small_synthetic_generation_pipeline": small_synthetic_generation_pipeline,
        "medium_synthetic_generation_pipeline": small_synthetic_generation_pipeline,
        "large_synthetic_generation_pipeline": small_synthetic_generation_pipeline,
        "support_generate_pipeline": support_generate_pipeline,
        "support_demo_pipeline": support_demo_pipeline,
        "small_end_to_end": small_end_to_end,
        "medium_end_to_end": medium_end_to_end,
        "large_end_to_end": large_end_to_end,
    }
