import numpy as np
from scipy.stats import norm
from scipy.stats import gmean
import pandas as pd

class OptionPricingEngine:
    def __init__(
        self, type, S0, T, t, ground_paths_df, gen_paths_df, 
        r = None, sigma = None, approx_exact = False, input_is_real_data = False
    ):
        self.type = type
        self.S0 = S0
        self.T = T
        self.t = t
        self.r = r
        self.sigma = sigma
        self.ground_paths_df = ground_paths_df
        self.gen_paths_df = gen_paths_df
        self.approx_exact = approx_exact
        self.input_is_real_data = input_is_real_data

        type_functions = {
            "European_exact": self.exact_european_prices,
            "European_call": self.european_call_payoff,
            "European_put": self.european_put_payoff,
            "Lookback_exact": self.exact_lookback_prices,
            "Lookback_call": self.lookback_call_payoff,
            "Lookback_put": self.lookback_put_payoff,
            "Asian_exact": self.exact_asian_prices,
            "Asian_call": self.asian_call_payoff,
            "Asian_put": self.asian_put_payoff,
        }
        if self.type+"_call" in type_functions:
            self.call_payoff = type_functions[self.type+"_call"]
            self.put_payoff = type_functions[self.type+"_put"]
            if not approx_exact and not input_is_real_data:
                self.exact_call_put = type_functions[self.type+"_exact"] 
        else:
            raise ValueError(f'Option type "{self.type}" currently not implemented.')

        
    ### European options ##############################################################################################
    def exact_european_prices(self, K):
        d1 = (np.log(self.S0/K) + (self.r + self.sigma**2/2)*(self.T-self.t)) / (self.sigma*np.sqrt(self.T-self.t))
        d2 = d1 - self.sigma*np.sqrt(self.T-self.t)
        exact_call = self.S0*norm.cdf(d1) - K*np.exp(-self.r*(self.T-self.t))*norm.cdf(d2)
        exact_put = K*np.exp(-self.r*(self.T-self.t)) - self.S0 + exact_call
        return exact_call, exact_put
    
    def european_call_payoff(self, x, K):
        return np.maximum(x.iloc[:, -1] - K, 0)
    
    def european_put_payoff(self, x, K):
        return np.maximum(K - x.iloc[:, -1], 0)
    
    ### Lookback options ##############################################################################################
    def exact_lookback_prices(self, T_cur):
        d = (self.r + self.sigma**2/2)*np.sqrt(T_cur-self.t)/self.sigma
        exact_call = self.S0*(
            norm.cdf(d) - 
            np.exp(-self.r*(T_cur-self.t))*norm.cdf(d-self.sigma*np.sqrt(T_cur-self.t)) - 
            self.sigma**2/(2*self.r)*norm.cdf(-d) +
            np.exp(-self.r*(T_cur-self.t))*self.sigma**2/(2*self.r)*norm.cdf(d-self.sigma*np.sqrt(T_cur-self.t))
        )
        exact_put = self.S0*(
            -norm.cdf(-d) + 
            np.exp(-self.r*(T_cur-self.t))*norm.cdf(-d+self.sigma*np.sqrt(T_cur-self.t)) +
            self.sigma**2/(2*self.r)*norm.cdf(d) -
            np.exp(-self.r*(T_cur-self.t))*self.sigma**2/(2*self.r)*norm.cdf(-d+self.sigma*np.sqrt(T_cur-self.t))
        )
        return exact_call, exact_put
    
    def lookback_call_payoff(self, x):
        return np.maximum(x.iloc[:, -1] - x.min(axis=1), 0)
    
    def lookback_put_payoff(self, x):
        return np.maximum(x.max(axis=1) - x.iloc[:, -1], 0)
    
    ### Asian options ################################################################################################
    def exact_asian_prices(self, K):
        M1 = (np.exp(self.r*self.T)-1)/(self.r*self.T)*self.S0
        M2 = (
            2*np.exp((2*self.r+self.sigma**2)*self.T)*self.S0**2 / ((self.r+self.sigma**2)*(2*self.r+self.sigma**2)*self.T**2) + 
            2*self.S0**2 / (self.r*self.T**2) * (1/(2*self.r+self.sigma**2) - np.exp(self.r*self.T)/(self.r+self.sigma**2))
        )
        sigma_tilde = np.sqrt(np.log(M2/M1**2)/self.T)
        d1 = (np.log(M1/K) + sigma_tilde**2*self.T/2) / (sigma_tilde*np.sqrt(self.T))
        d2 = d1 - sigma_tilde*np.sqrt(self.T)

        exact_call = np.exp(-self.r*self.T)*(M1*norm.cdf(d1) - K*norm.cdf(d2))
        exact_put = np.exp(-self.r*self.T)*(K*norm.cdf(-d2) - M1*norm.cdf(-d1))

        return exact_call, exact_put

    def asian_call_payoff(self, x, K):
        return np.maximum(x.apply(gmean, axis=1) - K, 0)
    
    def asian_put_payoff(self, x, K):
        return np.maximum(K - x.apply(gmean, axis=1), 0)

    ### Calculations #################################################################################################
    def calc_all_T(self, grid_size=1, approx_df=None, recalculate_all=False, resample=True, recalculate_input=False):
        self.calculate_option_prices_T(
            grid_size = grid_size, approx_df = approx_df, recalculate_all=recalculate_all, resample=resample, recalculate_input=recalculate_input
        )
        self.calculate_option_price_deviation_absolute()
        self.calculate_option_price_deviation_relative()
        return

    def calc_all_K(self, K_values=np.linspace(0.5, 1.5, 100), approx_df=None, recalculate_all=False, resample=True, recalculate_input=False):
        self.calculate_option_prices_K(
            K_values = K_values, approx_df = approx_df, recalculate_all=recalculate_all, resample=resample, recalculate_input=recalculate_input
        )
        self.calculate_option_price_deviation_absolute()
        self.calculate_option_price_deviation_relative()
        return

    def calculate_option_prices_T(self, grid_size=1, approx_df=None, recalculate_all=False, resample=True, recalculate_input=False):
        # Needed for lookback options
        # Store results
        self.T_values = []
        self.mc_call_gen_prices = []
        self.mc_put_gen_prices = []
        recalculate = (
            not (
                hasattr(self, "exact_call_prices") and hasattr(self, "exact_put_prices") and \
                hasattr(self, "mc_call_ground_prices") and hasattr(self, "mc_put_ground_prices")
            ) or recalculate_all
        ) and not self.input_is_real_data

        if recalculate or recalculate_input:
            if recalculate:
                self.exact_call_prices = []
                self.exact_put_prices = []
            self.mc_call_ground_prices = []
            self.mc_put_ground_prices = []

        for i, T_cur in enumerate(self.ground_paths_df.columns):
            if T_cur == 0 or (i % grid_size != 0 and T_cur != self.T):
                continue

            if resample:
                # if resample, take 1000 random paths/rows for new batch
                sample_df = self.gen_paths_df.sample(n=1000, axis=0, random_state=int(T_cur*10000))
                mc_call_gen_price = np.mean(np.exp(-self.r*(T_cur-self.t))*self.call_payoff(sample_df.iloc[:, :(i+1)]))
                mc_put_gen_price = np.mean(np.exp(-self.r*(T_cur-self.t))*self.put_payoff(sample_df.iloc[:, :(i+1)]))
            else:
                # take all paths/rows
                mc_call_gen_price = np.mean(np.exp(-self.r*(T_cur-self.t))*self.call_payoff(self.gen_paths_df.iloc[:, :(i+1)]))
                mc_put_gen_price = np.mean(np.exp(-self.r*(T_cur-self.t))*self.put_payoff(self.gen_paths_df.iloc[:, :(i+1)]))
            self.mc_call_gen_prices.append(mc_call_gen_price)
            self.mc_put_gen_prices.append(mc_put_gen_price)
            self.T_values.append(T_cur)
            
            if recalculate or recalculate_input:
                if recalculate:
                    if self.approx_exact:
                        exact_call_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.call_payoff(approx_df.iloc[:, :(i+1)]))
                        exact_put_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.put_payoff(approx_df.iloc[:, :(i+1)]))
                    else: 
                        exact_call_price, exact_put_price = self.exact_call_put(T_cur)

                    self.exact_call_prices.append(exact_call_price)
                    self.exact_put_prices.append(exact_put_price)

                mc_call_ground_price = np.mean(np.exp(-self.r*(T_cur-self.t))*self.call_payoff(self.ground_paths_df.iloc[:, :(i+1)]))
                mc_put_ground_price = np.mean(np.exp(-self.r*(T_cur-self.t))*self.put_payoff(self.ground_paths_df.iloc[:, :(i+1)]))
                
                self.mc_call_ground_prices.append(mc_call_ground_price)
                self.mc_put_ground_prices.append(mc_put_ground_price)

        self.results_df = pd.DataFrame({
            "T_values": self.T_values, 
            "mc_call_ground_prices": self.mc_call_ground_prices,
            "mc_put_ground_prices": self.mc_put_ground_prices,
            "mc_call_gen_prices": self.mc_call_gen_prices,
            "mc_put_gen_prices": self.mc_put_gen_prices
        })
        if not self.input_is_real_data:
            self.results_df["exact_call_prices"] = self.exact_call_prices
            self.results_df["exact_put_prices"] = self.exact_put_prices
        return

    def calculate_option_prices_K(self, K_values, approx_df=None, recalculate_all=False, resample=True, recalculate_input=False):
        # Store results
        self.K_values = K_values
        self.mc_call_gen_prices = []
        self.mc_put_gen_prices = []
        recalculate = (
            not (
                hasattr(self, "exact_call_prices") and hasattr(self, "exact_put_prices") and \
                hasattr(self, "mc_call_ground_prices") and hasattr(self, "mc_put_ground_prices")
            ) or recalculate_all
        ) and not self.input_is_real_data

        if recalculate or recalculate_input:
            if recalculate:
                self.exact_call_prices = []
                self.exact_put_prices = []
            self.mc_call_ground_prices = []
            self.mc_put_ground_prices = []

        for K in K_values:
            if resample:
                # if resample, take 1000 random paths/rows for new batch
                sample_df = self.gen_paths_df.sample(n=1000, axis=0, random_state=int(K*10000))
                mc_call_gen_price = np.mean(
                    np.exp(-self.r*(self.T-self.t))*self.call_payoff(sample_df, K)
                )
                mc_put_gen_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.put_payoff(sample_df, K))
            else:
                # take all paths/rows
                mc_call_gen_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.call_payoff(self.gen_paths_df, K))
                mc_put_gen_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.put_payoff(self.gen_paths_df, K))
        
            self.mc_call_gen_prices.append(mc_call_gen_price)
            self.mc_put_gen_prices.append(mc_put_gen_price)
            
            if recalculate or recalculate_input:
                if recalculate:
                    if self.approx_exact:
                        exact_call_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.call_payoff(approx_df, K))
                        exact_put_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.put_payoff(approx_df, K))
                    else: 
                        exact_call_price, exact_put_price = self.exact_call_put(K)
                    self.exact_call_prices.append(exact_call_price)
                    self.exact_put_prices.append(exact_put_price)

                mc_call_ground_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.call_payoff(self.ground_paths_df, K))
                mc_put_ground_price = np.mean(np.exp(-self.r*(self.T-self.t))*self.put_payoff(self.ground_paths_df, K))

                self.mc_call_ground_prices.append(mc_call_ground_price)
                self.mc_put_ground_prices.append(mc_put_ground_price)

        self.results_df = pd.DataFrame({
            "K_values": self.K_values, 
            "mc_call_ground_prices": self.mc_call_ground_prices,
            "mc_put_ground_prices": self.mc_put_ground_prices,
            "mc_call_gen_prices": self.mc_call_gen_prices,
            "mc_put_gen_prices": self.mc_put_gen_prices
        })
        if not self.input_is_real_data:
            self.results_df["exact_call_prices"] = self.exact_call_prices
            self.results_df["exact_put_prices"] = self.exact_put_prices
        return
    
    def calculate_option_price_deviation_absolute(self):
        if not self.input_is_real_data:
            # deviations from (approximate) exact prices
            self.ground_call_deviations = abs(np.array(self.mc_call_ground_prices) - np.array(self.exact_call_prices))
            self.ground_put_deviations = abs(np.array(self.mc_put_ground_prices) - np.array(self.exact_put_prices))
            self.gen_call_deviations = abs(np.array(self.mc_call_gen_prices) - np.array(self.exact_call_prices))
            self.gen_put_deviations = abs(np.array(self.mc_put_gen_prices) - np.array(self.exact_put_prices))
            new_columns = pd.DataFrame({
                "ground_call_deviations": self.ground_call_deviations,
                "ground_put_deviations": self.ground_put_deviations,
                "gen_call_deviations": self.gen_call_deviations,
                "gen_put_deviations": self.gen_put_deviations,
            })
        else:
            # deviations from input data prices
            self.gen_call_deviations = abs(np.array(self.mc_call_gen_prices) - np.array(self.mc_call_ground_prices))
            self.gen_put_deviations = abs(np.array(self.mc_put_gen_prices) - np.array(self.mc_put_ground_prices))
            new_columns = pd.DataFrame({
                "gen_call_deviations": self.gen_call_deviations,
                "gen_put_deviations": self.gen_put_deviations,
            })

        self.results_df = pd.concat([self.results_df, new_columns], axis=1)
        return
    
    def calculate_option_price_deviation_relative(self):
        if not self.input_is_real_data:
            self.ground_call_deviations_rel = self.ground_call_deviations / np.array(self.exact_call_prices) * 100
            self.ground_put_deviations_rel = self.ground_put_deviations / np.array(self.exact_put_prices) * 100
            self.gen_call_deviations_rel = self.gen_call_deviations / np.array(self.exact_call_prices) * 100
            self.gen_put_deviations_rel = self.gen_put_deviations / np.array(self.exact_put_prices) * 100
            new_columns = pd.DataFrame({
                "ground_call_deviations_rel": self.ground_call_deviations_rel,
                "ground_put_deviations_rel": self.ground_put_deviations_rel,
                "gen_call_deviations_rel": self.gen_call_deviations_rel,
                "gen_put_deviations_rel": self.gen_put_deviations_rel,
            })
        else:
            self.gen_call_deviations_rel = self.gen_call_deviations / np.array(self.mc_call_ground_prices) * 100
            self.gen_put_deviations_rel = self.gen_put_deviations / np.array(self.mc_put_ground_prices) * 100
            new_columns = pd.DataFrame({
                "gen_call_deviations_rel": self.gen_call_deviations_rel,
                "gen_put_deviations_rel": self.gen_put_deviations_rel,
            })

        self.results_df = pd.concat([self.results_df, new_columns], axis=1)
        return
