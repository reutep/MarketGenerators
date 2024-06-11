import matplotlib.pyplot as plt
class OptionPricingVisualization:
    def __init__(self, pricing_engine, exact_label="Theoretical ", file_name=None):
        self.pe = pricing_engine
        self.title_prefix = self.pe.type + " "
        self.exact_label = exact_label
        self.file_name = file_name
        if hasattr(self.pe, 'T_values'):
            self.x_values = self.pe.T_values
            self.x_label = "Maturity (T)"
            self.title_suffix = "Maturities (T)"
        else:
            self.x_values = self.pe.K_values
            self.x_label = "Strike Price (K)"
            self.title_suffix = "Strike Prices (K)"

    def plot_option_prices(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.pe.exact_call_prices, 'C2--', label=self.exact_label+'Call Price')
        plt.plot(self.x_values, self.pe.mc_call_ground_prices, 'C1--', label='Ground Truth Call Price')
        plt.plot(self.x_values, self.pe.mc_call_gen_prices, 'C0--', label='Generated Call Price')
        plt.plot(self.x_values, self.pe.exact_put_prices, 'C2:', label=self.exact_label+'Put Price')
        plt.plot(self.x_values, self.pe.mc_put_ground_prices, 'C1:', label='Ground Truth Put Price')
        plt.plot(self.x_values, self.pe.mc_put_gen_prices, 'C0:', label='Generated Put Price')

        plt.xlabel(self.x_label)
        plt.ylabel('Option Price')
        plt.title(self.title_prefix + 'Option Prices for Different ' + self.title_suffix)
        plt.legend()
        plt.grid(True)
        if self.file_name is not None:
            plt.savefig(f"plots/{self.title_prefix.strip().lower()}_option_prices_{self.file_name}")
        plt.show()
        return

    def plot_option_price_deviation(self):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.pe.ground_call_deviations, 'C1--' ,label='Ground Truth Call Deviation')
        plt.plot(self.x_values, self.pe.gen_call_deviations, 'C0--' ,label='Generated Call Deviation')
        plt.plot(self.x_values, self.pe.ground_put_deviations, 'C1:',label='Ground Truth Put Deviation')
        plt.plot(self.x_values, self.pe.gen_put_deviations, 'C0:',label='Generated Put Deviation')

        plt.xlabel(self.x_label)
        plt.ylabel(f'Deviation from {self.exact_label}Price')
        plt.title(f'Deviation of {self.title_prefix}Option Prices for Different ' + self.title_suffix)
        plt.legend()
        plt.grid(True)
        if self.file_name is not None:
            plt.savefig(f"plots/{self.title_prefix.strip().lower()}_option_price_devs_{self.file_name}")
        plt.show()       
        return
    
    def plot_option_price_deviation_relative(self, zoom_ylimits=None):
        plt.figure(figsize=(10, 6))
        plt.plot(self.x_values, self.pe.ground_call_deviations_rel, 'C1--' ,label='Ground Truth Call Deviation')
        plt.plot(self.x_values, self.pe.gen_call_deviations_rel, 'C0--' ,label='Generated Call Deviation')
        plt.plot(self.x_values, self.pe.ground_put_deviations_rel, 'C1:',label='Ground Truth Put Deviation')
        plt.plot(self.x_values, self.pe.gen_put_deviations_rel, 'C0:',label='Generated Put Deviation')

        plt.xlabel(self.x_label)
        plt.ylabel(f'Relative Deviation from {self.exact_label}Price (%)')
        plt.title(f'Relative Deviation of {self.title_prefix}Option Prices for Different ' + self.title_suffix)
        plt.legend()
        plt.ylim(zoom_ylimits)
        plt.grid(True)
        if self.file_name is not None:
            if zoom_ylimits is not None:
                plt.savefig(f"plots/{self.title_prefix.strip().lower()}_option_price_devs_rel_zoom_{self.file_name}")
            else:
                plt.savefig(f"plots/{self.title_prefix.strip().lower()}_option_price_devs_rel_{self.file_name}")
        plt.show()
        return
    