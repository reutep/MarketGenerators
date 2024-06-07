# import pandas as pd
import numpy as np

def print_basic_gbm_metrics(n_periods, annualization_factor, ground_paths_df, recovered_paths_df, exp_stdev, mu, return_threshold):
    recovered_log_returns_df = np.log(recovered_paths_df).diff(axis=1).iloc[:, 1:]
    ground_log_returns_df = np.log(ground_paths_df).diff(axis=1).iloc[:, 1:]
    
    # compute statistics
    generated_mean = np.mean(recovered_paths_df.iloc[:, -1])
    input_mean = np.mean(ground_paths_df.iloc[:, -1])
    expected_mean = np.exp(mu * n_periods)
    ann_gen_stdev = recovered_log_returns_df.std(axis=1).mean() * np.sqrt(annualization_factor)
    ann_pat_stdev = ground_log_returns_df.std(axis=1).mean() * np.sqrt(annualization_factor)
    ul_percentage = ((abs(ground_log_returns_df.values.flatten()) > return_threshold).sum() / len(ground_log_returns_df.values.flatten())) * 100
    gen_percentage = ((abs(recovered_log_returns_df.values.flatten()) > return_threshold).sum() / len(recovered_log_returns_df.values.flatten())) * 100

    print(f"Generated mean:\t {generated_mean:.5f}")
    print(f"Input mean:\t {input_mean:.5f}")
    print(f"Expected mean:\t {expected_mean:.5f}")
    print("-------------------------------------")
    print(f"Ann. gen stdev:\t {ann_gen_stdev:.5f}")
    print(f"Ann. pat stdev:\t {ann_pat_stdev:.5f}")
    print(f"Exp. stdev:\t {exp_stdev:.5f}")
    print("-------------------------------------")
    print(f"ul % > {return_threshold*100:.2f}%:\t {ul_percentage:.5f}")
    print(f"gen % > {return_threshold*100:.2f}%:\t {gen_percentage:.5f}")
    return