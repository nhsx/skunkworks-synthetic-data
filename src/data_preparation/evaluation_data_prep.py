import pandas as pd
import numpy as np

import io
from matplotlib import pyplot as plt
import base64


def select_columns(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, features_to_use: pd.DataFrame
) -> pd.DataFrame:
    real_data = real_data[features_to_use]
    synthetic_data = synthetic_data[features_to_use]
    return dict(real_data_columns=real_data, synthetic_data_columns=synthetic_data)


def combine_data(real_data: pd.DataFrame, synthetic_data: pd.DataFrame) -> pd.DataFrame:
    real_data["synthetic_data"] = 0
    synthetic_data["synthetic_data"] = 1
    data = real_data.append(synthetic_data)
    return data


def uncombine_data(data: pd.DataFrame) -> dict:
    real_data = data[data["synthetic_data"] == 0]
    synthetic_data = data[data["synthetic_data"] == 1]

    real_data.drop("synthetic_data", axis=1, inplace=True)
    synthetic_data.drop("synthetic_data", axis=1, inplace=True)
    return dict(real_data=real_data, synthetic_data=synthetic_data)


def convert_time_features(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, time_features: list
) -> dict:
    data = combine_data(real_data, synthetic_data)
    for column in time_features:
        data[column] = data[column].apply(
            lambda x: (pd.to_datetime(x, format="%Y-%m-%d %H:%M:%S").value)
            if x is not np.nan
            else np.nan
        )
    real_and_syn_data = uncombine_data(data)
    return real_and_syn_data


def convert_categorical_features(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame, categorical_features: list
) -> dict:
    data = combine_data(real_data, synthetic_data)
    data = pd.get_dummies(
        data, prefix=categorical_features, columns=categorical_features
    )
    real_and_syn_data = uncombine_data(data)
    return real_and_syn_data


def save_image_as_bytes() -> str:
    figure_bytes = io.BytesIO()
    plt.savefig(figure_bytes, format="png")
    figure_bytes.seek(0)
    return base64.b64encode(figure_bytes.read()).decode("utf-8")


def fill_missing_values_synthetic_real(
    real_data: pd.DataFrame, synthetic_data: pd.DataFrame
) -> pd.DataFrame:
    return dict(real_data=real_data.fillna(0), synthetic_data=synthetic_data.fillna(0))
