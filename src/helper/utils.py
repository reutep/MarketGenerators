from io import StringIO 
import sys
import numpy as np
import pandas as pd
from src.evaluate.option_pricing import OptionPricingEngine

# capture all outputs of print functions
class Capturing(list):
    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self
    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        del self._stringio # free up some memory
        sys.stdout = self._stdout


def load_input_and_generated_returns(input_file, generated_file, nDays, T): 
    all_input_returns = np.load(input_file).flatten()
    n_input_returns = np.prod(all_input_returns.shape)
    last_input_index = n_input_returns - (n_input_returns % nDays)
    input_returns = all_input_returns[:last_input_index].reshape((-1, nDays))
    input_prices_np = np.exp(np.cumsum(input_returns, axis=1))     
    input_prices_df = pd.DataFrame(np.insert(input_prices_np, 0, 1, axis=1))

    all_generated_returns = np.load(generated_file).flatten() 
    n_generated_returns = np.prod(all_generated_returns.shape)
    last_generated_index = n_generated_returns - (n_generated_returns % nDays)
    generated_returns = all_generated_returns[:last_generated_index].reshape((-1, nDays))

    generated_prices_np = np.exp(np.cumsum(generated_returns, axis=1))               
    generated_prices_df = pd.DataFrame(np.insert(generated_prices_np, 0, 1, axis=1))
    # Renaming of columns if mandatory in order to make Lookback option pricing work properly
    input_prices_df.columns = np.linspace(0, T, input_prices_df.shape[1])
    generated_prices_df.columns = np.linspace(0, T, generated_prices_df.shape[1])
    
    return input_prices_df, generated_prices_df, generated_returns


def initialize_all_option_engines(
        input_prices_df, generated_prices_df, T, input_is_real_data=False, t=0, S0=1, approx_exact=False
    ):
    # Initialize new european option pricing engine
    european_engine = OptionPricingEngine(
        type="European", 
        S0=S0, 
        T=T, 
        t=t, 
        ground_paths_df=input_prices_df, 
        gen_paths_df=generated_prices_df,
        input_is_real_data=input_is_real_data,
        approx_exact=approx_exact
    )
    # Initialize new asian option pricing engine
    asian_engine = OptionPricingEngine(
        type="Asian", 
        S0=S0, 
        T=T, 
        t=t, 
        ground_paths_df=input_prices_df, 
        gen_paths_df=generated_prices_df,
        input_is_real_data=input_is_real_data,
        approx_exact=approx_exact
    )
    # Initialize new lookback option pricing engine
    lookback_engine = OptionPricingEngine(
        type="Lookback", 
        S0=S0, 
        T=T, 
        t=t, 
        ground_paths_df=input_prices_df, 
        gen_paths_df=generated_prices_df,
        input_is_real_data=input_is_real_data,
        approx_exact=approx_exact
    )   
    return european_engine, asian_engine, lookback_engine

def fill_results(
        european_engine, asian_engine, lookback_engine, results_call, results_put, gen_model, 
        recalculate_input=True, has_input_dev=True
    ):
    for i, K in enumerate(european_engine.K_values):
        setting=f"European_K={K:.2f}"
        results_call[setting][gen_model].append(european_engine.gen_call_deviations[i])
        results_put[setting][gen_model].append(european_engine.gen_put_deviations[i])
        if recalculate_input:
            if has_input_dev:
                results_call[setting]["Input"].append(european_engine.ground_call_deviations[i])
                results_put[setting]["Input"].append(european_engine.ground_put_deviations[i])
            else:
                results_call[setting]["Input"].append(0.)
                results_put[setting]["Input"].append(0.)

        setting=f"Asian_K={K:.2f}"
        results_call[setting][gen_model].append(asian_engine.gen_call_deviations[i])
        results_put[setting][gen_model].append(asian_engine.gen_put_deviations[i])
        if recalculate_input:
            if has_input_dev:
                results_call[setting]["Input"].append(asian_engine.ground_call_deviations[i])
                results_put[setting]["Input"].append(asian_engine.ground_put_deviations[i])
            else:
                results_call[setting]["Input"].append(0.)
                results_put[setting]["Input"].append(0.)

    results_call["Lookback"][gen_model].append(lookback_engine.gen_call_deviations[0])
    results_put["Lookback"][gen_model].append(lookback_engine.gen_put_deviations[0])   
    if recalculate_input:
        if has_input_dev:
            results_call["Lookback"]["Input"].append(lookback_engine.ground_call_deviations[0])
            results_put["Lookback"]["Input"].append(lookback_engine.ground_put_deviations[0])
        else:
            results_call["Lookback"]["Input"].append(0.)
            results_put["Lookback"]["Input"].append(0.)
 
    return results_call, results_put


def save_stat_analysis_to_csv(
        settings, results, lookback_engine, european_engine, asian_engine,
        relevant_dir, target_subfolders, nDays, type_option="call"
    ):
    summary_data = []
    for i, setting in enumerate(settings):
        if setting.startswith("Lookback"):
            current_engine = lookback_engine
        elif setting.startswith("European"):
            current_engine = european_engine
        elif setting.startswith("Asian"):
            current_engine = asian_engine
        for gan_model in target_subfolders + ["Input"]:
            all_devs = np.array(results[setting][gan_model])
            average_dev = np.mean(all_devs)
            std_dev = np.std(all_devs)
            mc_ground_prices = getattr(current_engine, f"mc_{type_option}_ground_prices")()
            true_price = mc_ground_prices[i % len(european_engine.K_values)]
            summary_data.append([setting, gan_model, average_dev, std_dev, true_price])
    
    columns = ['Setting', 'Model', 'AverageDev', 'StdDev', 'TruePrice']
    summary_df = pd.DataFrame(summary_data, columns=columns)
    summary_df.to_csv(relevant_dir+f'/retrained_model_summary_{type_option}_NDays={nDays}.csv', index=False)