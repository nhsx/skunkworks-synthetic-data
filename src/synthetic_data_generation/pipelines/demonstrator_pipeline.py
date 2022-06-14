from functools import partial

from kedro.pipeline import Pipeline, node
from kedro.config import ConfigLoader

from data_preparation.generate_input_training_file import *
from synthetic_data_generation.train_generate_synthetic import *
from synthetic_data_generation.support_demo import *


def support_pipeline(**kwargs) -> Pipeline:
    nodes = []

    support_load = node(
        func=support_demo_load,
        inputs=None,
        outputs="support_input_data",
        name="support_load",
    )
    nodes.append(support_load)

    support_generate = node(
        func=support_demo_generation,
        inputs=["support_input_data", "params:synthetic_data_generation_size"],
        outputs="support_synthetic_data",
        name="support_generate",
    )
    nodes.append(support_generate)

    return Pipeline(nodes)
