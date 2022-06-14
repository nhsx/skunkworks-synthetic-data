import pandas as pd
import numpy as np
from sdv.tabular import CTGAN

data = pd.read_csv('../data/table_one.csv')
model = CTGAN(epochs=10)
model.fit(data)
synthetic_data = model.sample()
synthetic_data.to_csv("../data/synthetic_data_ct_gan1.csv")