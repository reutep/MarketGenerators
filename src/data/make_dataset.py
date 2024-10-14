import numpy as np
import pandas as pd
# needed for Python 3.7 and below
from typing import Union, Tuple, Dict
# fractional brownian motion
from fbm import FBM
# yahoo finance data
import yfinance as yf
class DataLoader:
    def __init__(self, method: str, params: Dict[str, Union[float, int]], seed: int = None):
        self.method_functions = {
            "Brownian_Motion": self.simulate_brownian_motion,
            "Fractional_BM": self.simulate_fractional_brownian_motion,
            "GBM": self.simulate_geometric_brownian_motion,
            "Kou_Jump_Diffusion": self.simulate_kou_jump_diffusion,
            "YFinance": self.get_yfinance_data
        }
        self.method = method
        self.params = params
        self.seed = seed

    def create_dataset(self, output_type: str = "DataFrame"):
        if self.method in self.method_functions:
            if self.seed is not None:
                np.random.seed(self.seed)
            paths, time = self.method_functions[self.method](**self.params)
            # Transform the data so that each time step is a row and each path is a column
            if output_type == "np.ndarray":
                return paths.T, time.T
            elif output_type == "DataFrame":
                return pd.DataFrame(paths, columns=time)
            else:
                raise ValueError(f'output_type={output_type} not implemented.')
        else:
            method_list = "', '".join(self.method_functions.keys())

            raise ValueError(
                f'Data creation method "{self.method}" currently not implemented. ' +
                f'Choose from "{method_list}".'
            )

        
    def simulate_brownian_motion(self, T: float, n_points: float, n: int):
        """
        Simulate n paths of scaled Brownian motion.

        Parameters:
        - T: Time horizon
        - n_points: Number of points in the time grid (including initial point)
        - n: Number of paths to simulate

        Returns:
        - A NumPy array of simulated scaled Brownian motion paths, shape (n, n_points)
        - A NumPy array of time steps
        """
        t = np.linspace(0, T, n_points)
        dt = T / (n_points - 1)
        dW = np.random.normal(size=(n, n_points-1))  # increments
        W = np.cumsum(np.sqrt(dt)*dW, axis=1)  # cumulative sum to generate paths
        W = np.hstack([np.zeros((n, 1)), W])  # Including zero at the start for the initial condition

        return W, t
    
    def simulate_fractional_brownian_motion(self, T: float, n_points: float, n: int, hurst: float):
        """
        Simulate n paths of fractional Brownian motion.

        Parameters:
        - T: Time horizon
        - n_points: Number of points in the time grid (including initial point)
        - n: Number of paths to simulate
        - hurst: Hurst parameter

        Returns:
        - A NumPy array of simulated fractional Brownian motion paths, shape (n, n_points)
        - A NumPy array of time steps
        """
        f = FBM(n=n_points-1, hurst=hurst, length=T, method="daviesharte")
        W = np.array([f.fbm() for _ in range(n)])
        t = f.times()

        return W, t

    def simulate_geometric_brownian_motion(
            self, S0: float, mu: float, sigma: float, T: float, n_points: float, n: int
            ):
        """
        Simulate n paths of the Black-Scholes process, each starting at S0.

        Parameters:
        - S0: Initial price
        - mu: Drift coefficient
        - sigma: Volatility
        - T: Time horizon
        - n_points: Number of points in the time grid (including initial point)
        - n: Number of paths to simulate

        Returns:
        - A NumPy array of simulated stock prices, shape (n, n_points)
        - A NumPy array of time steps
        """

        W, t = self.simulate_brownian_motion(T=T, n_points=n_points, n=n)

        # Calculate paths
        X = (mu - 0.5*sigma**2) * t + sigma * W
        S = S0 * np.exp(X)  # geometric brownian motion paths

        return S, t
    
    def simulate_kou_jump_diffusion(
            self, S0: float, mu: float, sigma: float, lambda_: float, p: float, eta1: float, eta2: float, 
            T: float, n_points: float, n: int):
        """
        Simulate n paths of the Kou jump-diffusion process, each starting at S0.

        Parameters:
        - S0: Initial price
        - mu: Drift coefficient
        - sigma: Volatility
        - lambda_: Jump intensity
        - p: Probability of positive jump
        - eta1: Rate of positive jump's exponential distribution
        - eta2: Rate of negative jump's exponential distribution
        - T: Time horizon
        - n_points: Number of points in the time grid (including initial point)
        - n: Number of paths to simulate

        Returns:
        - A NumPy array of simulated stock prices, shape (n, n_points)
        - A NumPy array of time steps
        """
        gbm, t = self.simulate_geometric_brownian_motion(S0=S0, mu=mu, sigma=sigma, T=T, n_points=n_points, n=n)
        
        # Jump component
        dv = np.ones((n, len(t)))
        # increments of Poisson process
        dt = T / (n_points - 1)
        dN = np.random.poisson(lambda_*dt, size=(n, len(t)-1))
        # TODO: (optional) Make this computation more efficient
        for i in range(n):
            for j in range(1, len(t)):
                # iterate through number of jumps
                # vi contains all jumps that happen in one increment
                vi = np.ones(dN[i, j-1])
                for k in range(dN[i, j-1]):
                    # loop over all jumps in one time step
                    gamma = np.random.exponential(1/eta1) if np.random.rand() < p else -np.random.exponential(1/eta2)
                    vi[k] = np.exp(gamma)
                # accumulate all jumps which happen in one timestep (1 if empty)
                dv[i, j] = np.prod(vi)
        # Calculate the paths with jumps
        S = gbm * np.cumprod(dv, axis=1) 
        return S, t
    
    def get_yfinance_data(self, S0: float=1., ticker="^GSPC", start=None, end="2024-06-30", n_points: int=22, split=False):
        """
        Download and reformat yfinance data starting at S0.

        Parameters:
        - S0: Initial price (rescale time series to start at S0)
        - T: Time horizon (in years)

        Returns:
        - A NumPy array of downloaded S&P500 data ending , shape (n, num_steps)
        - A NumPy array of time steps (rescaled such that first point is 0)
        """
        raw_data = yf.download(tickers=ticker, start=start, end=end, progress=False)["Adj Close"]
        S = np.array(raw_data)

        # if splt is True, split the data into chunks of length n_points:
        if split:
            returns = S[1:] / S[:-1]
            n = returns.shape[0] // (n_points-1)
            # split such that some data in beginning of the time series is lost (returns.shape[0] % (n_points-1) / (n_points-1))
            n_returns = n * (n_points-1)
            returns = returns[-n_returns:]
            returns = returns.reshape(n, n_points-1)
            # t is the annualized time between each return
            t_raw = np.array((raw_data.index[1:] - raw_data.index[0]).days)[-n_returns:]
            t_raw = t_raw.reshape(n, n_points-1)
            t = np.zeros((n, n_points))
            t[1:, 0] = t_raw[:-1,-1]
            t[:,1:] = t_raw - t[:, 0].reshape(-1, 1)
            t[:, 0] = 0
            t = t / 365.25  # convert days to years    
            S = np.zeros((n, n_points))
            S[:, 0] = S0
            S[:, 1:] = np.cumprod(returns, axis=1)
        else:
            S = S.reshape(1,-1)
            if S0 is not None:
                S = S / S[0,0] * S0
            t = np.array((raw_data.index - raw_data.index[0]).days)
            t = t / 365.25  # convert days to years

        return S, t

##### debug / test #####
if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    from src.features.data_transformer import Transformer

    brownian_motion_params = {
        "T": 5., 
        "n_points": 252, 
        "n": 1000
    }
    fractional_bm_params = {
        "T": 1., 
        "n_points": 252, 
        "n": 1000, 
        "hurst": 0.75
    }
    gbm_params = {
        "S0": 1., 
        "mu": 0.05,
        "sigma": 0.2, 
        "T": 5., 
        "n_points": 252, 
        "n": 1000
    }
    kou_params = {
        "S0": 1., 
        "mu": 0.12, 
        "sigma": 0.2, 
        "lambda_": 2.0, 
        "p": 0.3, 
        "eta1": 50., 
        "eta2": 25., 
        "T": 10., 
        "n_points": 252, 
        "n": 1000
    }
    bm_loader = DataLoader(method="Brownian_Motion", params=brownian_motion_params)
    gbm_loader = DataLoader(method="GBM", params=gbm_params)
    kou_loader = DataLoader(method="Kou_Jump_Diffusion", params=kou_params)
    prices_df = kou_loader.create_dataset(output_type="DataFrame")

    test = Transformer(prices_df)

    # Assuming df_times_as_index is the DataFrame with times as row indices
    plt.figure(figsize=(18, 10))
    for column in prices_df.columns:
        # plt.plot(prices_df.index, prices_df[column], alpha=1)  # for <= 50 paths
        # plt.plot(prices_df.index, prices_df[column], linewidth=0.1, alpha=0.4, color='blue') # for > 50 paths
        plt.plot(prices_df.index, prices_df[column], linewidth=0.1, alpha=0.2, color='blue') # for > 1000 paths

    plt.title('Simulated Paths over Time')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.ylim(0, 4)
    plt.grid(True)
    plt.show()
