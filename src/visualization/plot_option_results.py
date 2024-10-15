import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np
import pandas as pd
import src.visualization.set_plot_params

# Define the custom color cycle, starting from the second color
default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
shifted_color_cycle = default_color_cycle[1:] + default_color_cycle[:1]
double_shifted_color_cycle = default_color_cycle[2:] + default_color_cycle[:2]
plotDirName = "plots"

class OptionPricingVisualization:
    def __init__(self, pricing_engine, exact_label="Theoretical ", file_name=None):
        self.pe = pricing_engine
        self.title_prefix = self.pe.type + " "
        self.exact_label = exact_label
        self.file_name = file_name
        self.figCall, self.axCall, self.figPut, self.axPut = None, None, None, None
        self.figCallDev, self.axCallDev, self.figPutDev, self.axPutDev = None, None, None, None
        self.figCallDevRel, self.axCallDevRel, self.figPutDevRel, self.axPutDevRel = None, None, None, None

        if hasattr(self.pe, 'T_values'):
            self.x_values = self.pe.T_values
            self.x_label = "Maturity (T)"
        else:
            self.x_values = self.pe.K_values
            self.x_label = "Strike Price (K)"

    def set_y_limits_dev_abs(self, nDays, model_folder):
        """Set y-axis limits for absolute deviations."""
        y_lims = {
            'GBM': {
                5: {'European': 0.0072, 'Asian': 0.0037, 'Lookback': 0.019},
                10: {'European': 0.0125, 'Asian': 0.0059, 'Lookback': 0.0145},
                21: {'European': 0.026, 'Asian': 0.0135, 'Lookback': 0.0219},
                252: {'European': 0.34, 'Asian': 0.15, 'Lookback': 0.145}
            },
            'Kou_Jump_Diffusion': {
                5: {'European': 0.0075, 'Asian': 0.0037, 'Lookback': 0.0065},
                10: {'European': 0.013, 'Asian': 0.0075, 'Lookback': 0.0105},
                21: {'European': 0.032, 'Asian': 0.0165, 'Lookback': 0.0145},
                252: {'European': 0.38, 'Asian': 0.159, 'Lookback': 0.305}
            },
            'YFinance': {
                5: {'European': 0.073, 'Asian': 0.0038, 'Lookback': 0.0021},  # TBD
                10: {'European': 0.0126, 'Asian': 0.006, 'Lookback': 0.0146},  # TBD
                21: {'European': 0.028, 'Asian': 0.0136, 'Lookback': 0.022},  # TBD
                252: {'European': 0.33, 'Asian': 0.14, 'Lookback': 0.24}  # TBD
            }
        }

        if model_folder not in y_lims:
            raise ValueError(f"Unsupported model_folder: {model_folder}")
        
        if nDays not in y_lims[model_folder]:
            raise ValueError(f"Unsupported value for nDays: {nDays}")
        
        if self.pe.type not in y_lims[model_folder][nDays]:
            raise ValueError(f"Unsupported option type: {self.pe.type}")

        upper_limit = y_lims[model_folder][nDays][self.pe.type]
        
        return (0, upper_limit)

    def plot_option_prices(self, close=False, label="DUMMY_MODEL"):
        """Plot option prices."""
        if self.figCall is None and self.axCall is None:
            self.figCall, self.axCall = plt.subplots(figsize=(10, 6))
            self.axCall.set_xlabel(self.x_label)
            self.axCall.set_ylabel('Call Option Price')
            self.axCall.set_title(self.title_prefix + 'Call Option Prices')
            self.axCall.grid(True)
            if not self.pe.input_is_real_data:
                self.axCall.plot(self.x_values, self.pe.exact_call_prices, linestyle='--', label=self.exact_label)
            else:
                self.axCall.set_prop_cycle(cycler(color=shifted_color_cycle))
            self.axCall.plot(self.x_values, self.pe.mc_call_ground_prices, ':', label='Input Data')

        if self.figPut is None and self.axPut is None:
            self.figPut, self.axPut = plt.subplots(figsize=(10, 6))
            self.axPut.set_xlabel(self.x_label)
            self.axPut.set_ylabel('Put Option Price')
            self.axPut.set_title(self.title_prefix + 'Put Option Prices')
            self.axPut.grid(True)
            if not self.pe.input_is_real_data:
                self.axPut.plot(self.x_values, self.pe.exact_put_prices, linestyle='--', label=self.exact_label)
            else:
                self.axPut.set_prop_cycle(cycler(color=shifted_color_cycle))
            self.axPut.plot(self.x_values, self.pe.mc_put_ground_prices, ':', label='Input Data')
        
        self.axCall.plot(self.x_values, self.pe.mc_call_gen_prices, label=label)
        self.axPut.plot(self.x_values, self.pe.mc_put_gen_prices, label=label)

        if close:
            self.axCall.legend()
            self.axPut.legend()
            if self.file_name is not None:
                self.figCall.savefig(
                    f"{plotDirName}/{self.title_prefix.strip().lower()}_call_option_prices_{self.file_name}"
                )
                self.figPut.savefig(
                    f"{plotDirName}/{self.title_prefix.strip().lower()}_put_option_prices_{self.file_name}"
                )
            self.figCall.show()
            self.figPut.show()
            plt.close(self.figCall)
            plt.close(self.figPut)
            self.figPut, self.axPut, self.figCall, self.axCall = None, None, None, None
        return

    def plot_option_price_deviation(self, close=False, label="DUMMY_MODEL", zoom_ylimits=None):
        """Plot option price deviations."""
        relevant_col_cycle = double_shifted_color_cycle if self.pe.input_is_real_data else shifted_color_cycle
        if self.figCallDev is None and self.axCallDev is None:
            self.figCallDev, self.axCallDev = plt.subplots(figsize=(10, 6))
            self.axCallDev.set_prop_cycle(cycler(color=relevant_col_cycle))
            self.axCallDev.set_xlabel(self.x_label)
            self.axCallDev.set_ylabel(f'Dev. from {self.exact_label}Price')
            self.axCallDev.set_title(f'Deviation of {self.title_prefix}Call Option Prices')
            self.axCallDev.grid(True)
            self.axCallDev.set_ylim(zoom_ylimits)
            if not self.pe.input_is_real_data:
                self.axCallDev.plot(self.x_values, self.pe.ground_call_deviations, ':' ,label='Input Data')

        if self.figPutDev is None and self.axPutDev is None:
            self.figPutDev, self.axPutDev = plt.subplots(figsize=(10, 6))
            self.axPutDev.set_prop_cycle(cycler(color=relevant_col_cycle))
            self.axPutDev.set_xlabel(self.x_label)
            self.axPutDev.set_ylabel(f'Dev. from {self.exact_label}Price')
            self.axPutDev.set_title(f'Deviation of {self.title_prefix}Put Option Prices')
            self.axPutDev.grid(True)
            self.axPutDev.set_ylim(zoom_ylimits)
            if not self.pe.input_is_real_data:
                self.axPutDev.plot(self.x_values, self.pe.ground_put_deviations, ':' ,label='Input Data')
        
        self.axCallDev.plot(self.x_values, self.pe.gen_call_deviations, label=label)
        self.axPutDev.plot(self.x_values, self.pe.gen_put_deviations, label=label)

        if close:
            self.axCallDev.legend()
            self.axPutDev.legend()
            if self.file_name is not None:
                self.figCallDev.savefig(
                    f"{plotDirName}/{self.title_prefix.strip().lower()}_call_option_dev_{self.file_name}"
                )
                self.figPutDev.savefig(
                    f"{plotDirName}/{self.title_prefix.strip().lower()}_put_option_dev_{self.file_name}"
                )
            self.figCallDev.show()
            self.figPutDev.show()
            plt.close(self.figCallDev)
            plt.close(self.figPutDev)
            self.figCallDev, self.axCallDev, self.figPutDev, self.axPutDev = None, None, None, None
        return
    
    def plot_option_price_deviation_relative(self, close=False, label="DUMMY_MODEL", zoom_ylimits=None):
        """Plot relative option price deviations."""
        relevant_col_cycle = double_shifted_color_cycle if self.pe.input_is_real_data else shifted_color_cycle
        if self.figCallDevRel is None and self.axCallDevRel is None:
            self.figCallDevRel, self.axCallDevRel = plt.subplots(figsize=(10, 6))
            self.axCallDevRel.set_prop_cycle(cycler(color=relevant_col_cycle))
            self.axCallDevRel.set_xlabel(self.x_label)
            self.axCallDevRel.set_ylabel(f'Rel. Dev. from {self.exact_label}Price (%)')
            self.axCallDevRel.set_title(f'Relative Deviation of {self.title_prefix}Call Option Prices')
            self.axCallDevRel.grid(True)
            if not self.pe.input_is_real_data:
                self.axCallDevRel.plot(self.x_values, self.pe.ground_call_deviations_rel, ':' ,label='Input Data')

        if self.figPutDevRel is None and self.axPutDevRel is None:
            self.figPutDevRel, self.axPutDevRel = plt.subplots(figsize=(10, 6))
            self.axPutDevRel.set_prop_cycle(cycler(color=relevant_col_cycle))
            self.axPutDevRel.set_xlabel(self.x_label)
            self.axPutDevRel.set_ylabel(f'Rel. Dev. from {self.exact_label}Price (%)')
            self.axPutDevRel.set_title(f'Relative Deviation of {self.title_prefix}Put Option Prices')
            self.axPutDevRel.grid(True)
            if not self.pe.input_is_real_data:
                self.axPutDevRel.plot(self.x_values, self.pe.ground_put_deviations_rel, ':' ,label='Input Data')
        
        self.axCallDevRel.plot(self.x_values, self.pe.gen_call_deviations_rel, label=label)
        self.axPutDevRel.plot(self.x_values, self.pe.gen_put_deviations_rel, label=label)

        if close:
            self.axCallDevRel.legend()
            self.axPutDevRel.legend()
            if self.file_name is not None:
                self.figCallDevRel.savefig(
                    f"{plotDirName}/{self.title_prefix.strip().lower()}_call_option_dev_rel_{self.file_name}"
                )
                self.figPutDevRel.savefig(
                    f"{plotDirName}/{self.title_prefix.strip().lower()}_put_option_dev_rel_{self.file_name}"
                )
            self.figCallDevRel.show()
            self.figPutDevRel.show()
            if zoom_ylimits is not None:
                self.axCallDevRel.set_ylim(zoom_ylimits)
                self.axPutDevRel.set_ylim(zoom_ylimits)
                if self.file_name is not None:
                    self.figCallDevRel.savefig(
                        f"{plotDirName}/{self.title_prefix.strip().lower()}_call_option_dev_rel_zoom_{self.file_name}"
                    )
                    self.figPutDevRel.savefig(
                        f"{plotDirName}/{self.title_prefix.strip().lower()}_put_option_dev_rel_zoom_{self.file_name}"
                    )
                self.figCallDevRel.show()
                self.figPutDevRel.show()
            plt.close(self.figCallDevRel)
            plt.close(self.figPutDevRel)
            self.figCallDevRel, self.axCallDevRel, self.figPutDevRel, self.axPutDevRel = None, None, None, None
        return

def option_csv_plotting(df, strike, option_type, price_label, file_name, bar_width=0.25, ylim=None):
    """Plot option data from CSV."""
    settings = ['European', 'Asian', 'Lookback']
    default_colors = plt.rcParams['axes.prop_cycle'].by_key()['color']  # Default Matplotlib colors

    df = df[df['Model'] != 'RCWGAN']

    european_df = df[(df['Setting'] == f"European_K={strike}")]
    asian_df = df[(df['Setting'] == f"Asian_K={strike}")]
    lookback_df = df[(df['Setting'] == "Lookback")]

    filtered_df = pd.concat([european_df, asian_df, lookback_df])
    models = filtered_df['Model'].unique()

    index = np.arange(len(models))  # Base position for each model group
    positions = [index + i * bar_width for i in range(3)]

    fig, ax = plt.subplots(figsize=(10, 6))

    for i, (setting, color) in enumerate(zip(settings, default_colors)):
        setting_df = filtered_df[filtered_df['Setting'].str.contains(setting)]
        ax.bar(
            positions[i], 
            setting_df['AverageDev'] / setting_df['TruePrice'] * 100, 
            bar_width,
            color=color, 
            label=f"{setting} {option_type.capitalize()}", 
            yerr=setting_df['StdDev'] / setting_df['TruePrice'] * 100, 
            capsize=5
        )

    ax.set_ylabel(f'Average Dev. from {price_label} Price (%)')
    ax.set_ylim(ylim)
    ax.set_xticks(index + bar_width)
    ax.set_xticklabels(models)
    ax.legend()
    ax.grid(axis="y")
    plt.tight_layout()
    plt.savefig(f"{file_name}.png")
    plt.close()