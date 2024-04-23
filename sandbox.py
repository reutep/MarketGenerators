# .\.venv\Scripts\activate 
from src.features import encoder 
from src.data import make_dataset
import numpy as np
import pandas as pd

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
prices_df = gbm_loader.create_dataset()

transformer = encoder.Transformer(prices_df)
pct_returns = transformer.calculate_daily_rolling_returns(5)
log_returns = transformer.calculate_daily_rolling_log_returns(5)
log_returns_reduced = transformer.calculate_returns(shift=5, logFlag=True)
pct_returns_reduced = transformer.calculate_returns(shift=5, logFlag=False)
print("####")