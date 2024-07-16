import matplotlib.pyplot as plt
from cycler import cycler

# Define the custom color cycle, starting from the second color
default_color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
shifted_color_cycle = default_color_cycle[1:] + default_color_cycle[:1]
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
            self.title_suffix = "Maturities (T)"
        else:
            self.x_values = self.pe.K_values
            self.x_label = "Strike Price (K)"
            self.title_suffix = "Strike Prices (K)"

    def plot_option_prices(self, close=False, label="DUMMY_MODEL"):
        # initialize plots
        if self.figCall is None and self.axCall is None:
            self.figCall, self.axCall = plt.subplots(figsize=(10, 6))
            self.axCall.set_xlabel(self.x_label, fontsize=14)
            self.axCall.set_ylabel('Call Option Price', fontsize=14)
            self.axCall.set_title(self.title_prefix + 'Call Option Prices for Different ' + self.title_suffix, fontsize=16)
            self.axCall.grid(True)
            self.axCall.plot(self.x_values, self.pe.exact_call_prices, linestyle='--', label=self.exact_label)
            self.axCall.plot(self.x_values, self.pe.mc_call_ground_prices, ':', label='Input Data')

        if self.figPut is None and self.axPut is None:
            self.figPut, self.axPut = plt.subplots(figsize=(10, 6))
            self.axPut.set_xlabel(self.x_label, fontsize=14)
            self.axPut.set_ylabel('Put Option Price', fontsize=14)
            self.axPut.set_title(self.title_prefix + 'Put Option Prices for Different ' + self.title_suffix, fontsize=16)
            self.axPut.grid(True)
            self.axPut.plot(self.x_values, self.pe.exact_put_prices, linestyle='--', label=self.exact_label)
            self.axPut.plot(self.x_values, self.pe.mc_put_ground_prices, ':', label='Input Data')
        
        # add Monte Carlo generated prices 
        self.axCall.plot(self.x_values, self.pe.mc_call_gen_prices, label=label)
        self.axPut.plot(self.x_values, self.pe.mc_put_gen_prices, label=label)

        # Finalize plots
        if close:
            self.axCall.legend(fontsize=12)
            self.axPut.legend(fontsize=12)
            if self.file_name is not None:
                self.figCall.savefig(f"plots/{self.title_prefix.strip().lower()}_call_option_prices_{self.file_name}")
                self.figPut.savefig(f"plots/{self.title_prefix.strip().lower()}_put_option_prices_{self.file_name}")
            self.figCall.show()
            self.figPut.show()
            plt.close(self.figCall)
            plt.close(self.figPut)
            # reset the figure and axis
            self.figPut, self.axPut, self.figCall, self.axCall = None, None, None, None
        return

    def plot_option_price_deviation(self, close=False, label="DUMMY_MODEL"):
        # initialize plots
        if self.figCallDev is None and self.axCallDev is None:
            self.figCallDev, self.axCallDev = plt.subplots(figsize=(10, 6))
            # shift the color cycle by one to match the price plot colors
            self.axCallDev.set_prop_cycle(cycler(color=shifted_color_cycle))
            self.axCallDev.set_xlabel(self.x_label, fontsize=14)
            self.axCallDev.set_ylabel(f'Deviation from {self.exact_label}Price', fontsize=14)
            self.axCallDev.set_title(
                f'Deviation of {self.title_prefix}Call Option Prices for Different ' + self.title_suffix, fontsize=16
            )
            self.axCallDev.grid(True)
            self.axCallDev.plot(self.x_values, self.pe.ground_call_deviations, ':' ,label='Input Data')

        if self.figPutDev is None and self.axPutDev is None:
            self.figPutDev, self.axPutDev = plt.subplots(figsize=(10, 6))
            # shift the color cycle by one to match the price plot colors
            self.axPutDev.set_prop_cycle(cycler(color=shifted_color_cycle))
            self.axPutDev.set_xlabel(self.x_label, fontsize=14)
            self.axPutDev.set_ylabel(f'Deviation from {self.exact_label}Price', fontsize=14)
            self.axPutDev.set_title(
                f'Deviation of {self.title_prefix}Put Option Prices for Different ' + self.title_suffix, fontsize=16
            )
            self.axPutDev.grid(True)
            self.axPutDev.plot(self.x_values, self.pe.ground_put_deviations, ':' ,label='Input Data')
        
        # add Monte Carlo generated prices 
        self.axCallDev.plot(self.x_values, self.pe.gen_call_deviations, label=label)
        self.axPutDev.plot(self.x_values, self.pe.gen_put_deviations, label=label)

        # Finalize plots
        if close:
            self.axCallDev.legend(fontsize=12)
            self.axPutDev.legend(fontsize=12)
            if self.file_name is not None:
                self.figCallDev.savefig(f"plots/{self.title_prefix.strip().lower()}_call_option_dev_{self.file_name}")
                self.figPutDev.savefig(f"plots/{self.title_prefix.strip().lower()}_put_option_dev_{self.file_name}")
            self.figCallDev.show()
            self.figPutDev.show()
            plt.close(self.figCallDev)
            plt.close(self.figPutDev)
            # reset the figure and axis
            self.figCallDev, self.axCallDev, self.figPutDev, self.axPutDev = None, None, None, None
        return
    
    def plot_option_price_deviation_relative(self, close=False, label="DUMMY_MODEL", zoom_ylimits=None):
        # initialize plots
        if self.figCallDevRel is None and self.axCallDevRel is None:
            self.figCallDevRel, self.axCallDevRel = plt.subplots(figsize=(10, 6))
            # shift the color cycle by one to match the price plot colors
            self.axCallDevRel.set_prop_cycle(cycler(color=shifted_color_cycle))
            self.axCallDevRel.set_xlabel(self.x_label, fontsize=14)
            self.axCallDevRel.set_ylabel(f'Rel. Deviation from {self.exact_label}Price (%)', fontsize=14)
            self.axCallDevRel.set_title(
                f'Relative Deviation of {self.title_prefix}Call Option Prices for Different ' + self.title_suffix, fontsize=16
            )
            self.axCallDevRel.grid(True)
            self.axCallDevRel.plot(self.x_values, self.pe.ground_call_deviations_rel, ':' ,label='Input Data')

        if self.figPutDevRel is None and self.axPutDevRel is None:
            self.figPutDevRel, self.axPutDevRel = plt.subplots(figsize=(10, 6))
            # shift the color cycle by one to match the price plot colors
            self.axPutDevRel.set_prop_cycle(cycler(color=shifted_color_cycle))
            self.axPutDevRel.set_xlabel(self.x_label, fontsize=14)
            self.axPutDevRel.set_ylabel(f'Rel. Deviation from {self.exact_label}Price (%)', fontsize=14)
            self.axPutDevRel.set_title(
                f'Relative Deviation of {self.title_prefix}Put Option Prices for Different ' + self.title_suffix, fontsize=16
            )
            self.axPutDevRel.grid(True)
            self.axPutDevRel.plot(self.x_values, self.pe.ground_put_deviations_rel, ':' ,label='Input Data')
        
        # add Monte Carlo generated prices 
        self.axCallDevRel.plot(self.x_values, self.pe.gen_call_deviations_rel, label=label)
        self.axPutDevRel.plot(self.x_values, self.pe.gen_put_deviations_rel, label=label)

        # Finalize plots
        if close:
            self.axCallDevRel.legend(fontsize=12)
            self.axPutDevRel.legend(fontsize=12)
            if self.file_name is not None:
                self.figCallDevRel.savefig(f"plots/{self.title_prefix.strip().lower()}_call_option_dev_rel_{self.file_name}")
                self.figPutDevRel.savefig(f"plots/{self.title_prefix.strip().lower()}_put_option_dev_rel_{self.file_name}")
            self.figCallDevRel.show()
            self.figPutDevRel.show()
            if zoom_ylimits is not None:
                self.axCallDevRel.set_ylim(zoom_ylimits)
                self.axPutDevRel.set_ylim(zoom_ylimits)
                if self.file_name is not None:
                    self.figCallDevRel.savefig(f"plots/{self.title_prefix.strip().lower()}_call_option_dev_rel_zoom_{self.file_name}")
                    self.figPutDevRel.savefig(f"plots/{self.title_prefix.strip().lower()}_put_option_dev_rel_zoom_{self.file_name}")
                self.figCallDevRel.show()
                self.figPutDevRel.show()
            plt.close(self.figCallDevRel)
            plt.close(self.figPutDevRel)
            # reset the figure and axis
            self.figCallDevRel, self.axCallDevRel, self.figPutDevRel, self.axPutDevRel = None, None, None, None
        return
 