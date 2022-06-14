from functools import partial

from kedro.pipeline import Pipeline, node
from kedro.config import ConfigLoader

from data_preparation.generate_input_training_file import *
from synthetic_data_generation.train_generate_synthetic import *


def load_data_pipeline(**kwargs) -> Pipeline:
    nodes = []

    dl = node(
        func=load_mimic_data,
        inputs=None,
        outputs=[
            "Admissions",
            "Chartevents",
            "ICUstays",
            "Items",
            "Outputevents",
            "Patients",
        ],
        name="load_mimic",
    )
    nodes.append(dl)

    ppp = node(
        func=preproc_patients,
        inputs=["Patients"],
        outputs="procPatients",
        name="preproc_pats",
    )
    nodes.append(ppp)

    return Pipeline(nodes)


def small_preproc_pipeline(**kwargs) -> Pipeline:

    gen = node(
        func=generate_11k_dataset,
        inputs=["Admissions", "procPatients", "ICUstays", "Chartevents", "Items"],
        outputs="table_one_11040",
        name="generate_small_input_dataset",
    )

    return Pipeline([gen])


def medium_preproc_pipeline(**kwargs) -> Pipeline:

    gen = node(
        func=generate_81k_dataset,
        inputs=["Admissions", "procPatients", "ICUstays", "Chartevents", "Items"],
        outputs="table_one_imbalanced_81795",
        name="generate_medium_input_dataset",
    )

    return Pipeline([gen])


def large_preproc_pipeline(**kwargs) -> Pipeline:

    gen = node(
        func=generate_217k_dataset,
        inputs=["Admissions", "procPatients", "ICUstays", "Chartevents", "Items"],
        outputs="table_one_imbalanced_217010",
        name="generate_large_input_dataset",
    )

    return Pipeline([gen])


def generate_synthetic_input_pipeline(table_size: str, **kwargs) -> Pipeline:

    gen_input = node(
        func=train_model_generate_synthetic,
        inputs=[
            table_size,
            "params:synthetic_data_generation_size",
            "params:all_features",
            "params:categorical_features",
        ],
        outputs="synthetic_data_input",
        name="generate_synthetic_dataset",
    )

    return Pipeline([gen_input])
