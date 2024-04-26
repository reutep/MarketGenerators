# .\.venv\Scripts\activate 
from src.features import data_transformer 
from src.data import make_dataset
import numpy as np
import pandas as pd
from esig import tosig as ts

brownian_motion_params = {
    "T": 5., 
    "dt": 0.004, 
    "n": 1000
}
gbm_params = {
    "S0": 1., 
    "mu": 0.05,
    "sigma": 0.2, 
    "T": 5., 
    "dt": 0.004, 
    "n": 1000
}
kou_params = {
    "S0": 1., 
    "mu": 0.05, 
    "sigma": 0.16, 
    "lambda_": 1.0, 
    "p": 0.4, 
    "eta1": 10., 
    "eta2": 5., 
    "T": 5., 
    "dt": 0.004, 
    "n": 1000
}
bm_loader = make_dataset.DataLoader(method="Brownian_Motion", params=brownian_motion_params)
gbm_loader = make_dataset.DataLoader(method="GBM", params=gbm_params)
kou_loader = make_dataset.DataLoader(method="Kou_Jump_Diffusion", params=kou_params)
prices, time = kou_loader.create_dataset(output_type="np.ndarray")
prices_df = kou_loader.create_dataset(output_type="DataFrame")

transformer = data_transformer.Transformer(paths_df=prices_df, paths=prices, time=time)
signatures = transformer.calculate_signature(depth=5, type="normal")
log_signatures = transformer.calculate_signature(depth=5, type="log")
pct_returns = transformer.calculate_daily_rolling_returns_df(5)
log_returns = transformer.calculate_daily_rolling_log_returns_df(5)
log_returns_reduced = transformer.calculate_returns_df(shift=5, logFlag=True)
pct_returns_reduced = transformer.calculate_returns_df(shift=5, logFlag=False)
print("####")