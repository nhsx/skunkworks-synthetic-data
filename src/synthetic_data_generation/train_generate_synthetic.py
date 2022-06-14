from re import S
import warnings
from xmlrpc.client import Boolean
# Standard imports
import numpy as np
import pandas as pd
import torch

# For data preprocessing
from rdt import HyperTransformer
from sklearn.preprocessing import StandardScaler
from sklearn_pandas import DataFrameMapper

from synthetic_data_generation.SynthVAE.opacus.utils.uniform_sampler import UniformWithReplacementSampler

# For VAE dataset formatting
from torch.utils.data import TensorDataset, DataLoader

# VAE functions
from synthetic_data_generation.SynthVAE.VAE import Decoder, Encoder, VAE

# Other
from synthetic_data_generation.SynthVAE.utils import set_seed

def train_model_generate_synthetic(table_one: pd.DataFrame, synthetic_data_generation_size: int,
                                    all_columns: list, cat_columns: list) -> pd.DataFrame:


    warnings.filterwarnings("ignore")
    set_seed(0)

    my_seed = np.random.randint(1e6)
    diff_priv = False

    table_one['SUBJECT_ID'] = table_one['SUBJECT_ID'].astype(float)
    table_one['ADMITTIME'] = pd.to_datetime(table_one['ADMITTIME'])
    table_one = table_one[(table_one['ADMITTIME']>pd.Timestamp("1970-01-01"))]
    table_one['ADMITTIME'] = (table_one['ADMITTIME'] - pd.Timestamp("1970-01-01")) // pd.Timedelta('1s')
    table_one['ADMITTIME'] = table_one['ADMITTIME'].astype(float)
    table_one['age'] = np.abs(table_one['age'])

    # Explicitly type the categorical variables of the SUPPORT dataset
    table_cols = all_columns
    table_one = table_one[table_cols]
    ###############################################################################
    # DATA PREPROCESSING #
    # We one-hot the categorical cols and standardise the continuous cols
    print("Beginning data preprocessing")

    cat_cols = cat_columns
    cont_cols = list(set(all_columns)-set(cat_columns))

    transformer_dtypes = {
        "i": "one_hot_encoding",
        "f": "numerical",
        "O": "one_hot_encoding",
        "b": "one_hot_encoding",
        "M": "datetime",
    }
    ht = HyperTransformer(dtype_transformers=transformer_dtypes)
    ht.fit(table_one)
    transformed = ht.transform(table_one)

    print("Data transformed")
    num_categories = (
        np.array([len(table_one[col].unique()) for col in cat_cols])
    ).astype(int)
    num_continuous = len(cont_cols)
    cols_standardize = transformed.columns[0:num_continuous]
    cols_leave = transformed.columns[num_continuous:]
    standardize = [([col], StandardScaler()) for col in cols_standardize]
    leave = [([col], None) for col in cols_leave]
    x_mapper = DataFrameMapper(leave + standardize)
    x_train_df = x_mapper.fit_transform(transformed)
    x_train_df = x_mapper.transform(transformed)
    x_train = x_train_df.astype("float32")


    ###############################################################################
    # Prepare data for interaction with torch VAE
    Y = torch.Tensor(x_train)
    dataset = TensorDataset(Y)
    batch_size = 32

    generator = None
    sample_rate = batch_size / len(dataset)
    data_loader = DataLoader(
        dataset,
        batch_sampler=UniformWithReplacementSampler(
            num_samples=len(dataset), sample_rate=sample_rate, generator=generator
        ),
        pin_memory=True,
        generator=generator,
    )

    target_delta = 1e-3
    target_eps = 10.0

    diff_priv_in = ""
    if diff_priv:
        diff_priv_in = " with differential privacy"

    print(
        f"Train + Generate + Evaluate VAE{diff_priv_in}"
    )
    set_seed(my_seed)

    # Create VAE
    latent_dim = 2
    encoder = Encoder(x_train.shape[1], latent_dim)
    decoder = Decoder(
        latent_dim, num_continuous, num_categories=num_categories
    )
    vae = VAE(encoder, decoder)
    num_epochs = 30
    if diff_priv:
        vae.diff_priv_train(
            data_loader,
            n_epochs=50,
            C=50,
            target_eps=target_eps,
            target_delta=target_delta,
            sample_rate=sample_rate,
        )
        print(f"(epsilon, delta): {vae.get_privacy_spent(target_delta)}")
    else:
        vae.train(data_loader, n_epochs=num_epochs)

    print("Training complete, producing synthetic data")

    samples_ = vae.generate(synthetic_data_generation_size).detach().numpy()

    print("Synthetic data generated")

    samples = np.ones_like(samples_)
    samples[:, :num_continuous] = samples_[:, -num_continuous:]
    samples[:, num_continuous:] = samples_[:, :-num_continuous]
    samples = pd.DataFrame(samples)
    samples.columns = transformed.columns
    samples = ht.reverse_transform(samples)
    samples[cat_cols] = samples[cat_cols].astype(object)


    for feature in x_mapper.features:
        if feature[0][0] in cont_cols:
            f = feature[0][0]
            samples[f] = feature[1].inverse_transform(samples[f])

    print("Saving down synthetic data")
    samples['ROW_ID'] = np.arange(samples.shape[0])
    samples['SUBJECT_ID'] = np.round(samples['SUBJECT_ID'],0)
    samples['ADMITTIME'] = pd.to_datetime(samples['ADMITTIME'], unit='s', origin='unix') 
    final_column_names = ['ROW_ID'] + all_columns
    samples_final = samples[final_column_names]
    
    
    if (table_one.shape[0]) < 75000:
        samples_final.to_csv(f'data/07_model_output/synthetic_data_small_input.csv',index=False)
    elif (table_one.shape[0]) < 125000:
        samples_final.to_csv(f'data/07_model_output/synthetic_data_medium_input.csv',index=False)
    elif (table_one.shape[0]) < 200000:
        samples_final.to_csv(f'data/07_model_output/synthetic_data_large_input.csv',index=False)
    return samples_final