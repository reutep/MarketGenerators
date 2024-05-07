import numpy as np
import pandas as pd
# needed for Python 3.7 and below
from typing import Union, Tuple, Dict
from numpy.typing import NDArray


class DataLoader:
    def __init__(self, method: str, params: Dict[str, Union[float, int]], seed: int = None):
        self.method_functions = {
            "Brownian_Motion": self.simulate_brownian_motion,
            "GBM": self.simulate_geometric_brownian_motion,
            "Kou_Jump_Diffusion": self.simulate_kou_jump_diffusion
        }
        self.method = method
        self.params = params
        self.seed = seed

    def create_dataset(self, output_type: str = "DataFrame") -> Union[pd.DataFrame, Tuple[NDArray, NDArray]]:
        if self.method in self.method_functions:
            np.random.seed(self.seed)
            paths, time = self.method_functions[self.method](**self.params)
            # Transform the data so that each time step is a row and each path is a column
            if output_type == "np.ndarray":
                return paths.T, time
            elif output_type == "DataFrame":
                return pd.DataFrame(paths.T, index=time)
            else:
                raise ValueError(f'output_type={output_type} not implemented.')
        else:
            method_list = "', '".join(self.method_functions.keys())

            raise ValueError(
                f'Data creation method "{self.method}" currently not implemented. ' +
                f'Choose from "{method_list}".'
            )

        
    def simulate_brownian_motion(self, T: float, dt: float, n: int) -> Tuple[NDArray, NDArray]:
        """
        Simulate n paths of scaled Brownian motion.

        Parameters:
        - T: Time horizon
        - dt: Time step size
        - n: Number of paths to simulate

        Returns:
        - A NumPy array of simulated scaled Brownian motion paths, shape (n, num_steps+1)
        - A NumPy array of time steps
        """
        num_steps = int(T / dt)
        t = np.linspace(0, T, num_steps+1)
        dW = np.random.normal(size=(n, num_steps))  # scaled increments
        W = np.cumsum(np.sqrt(dt)*dW, axis=1)  # cumulative sum to generate paths
        W = np.hstack([np.zeros((n, 1)), W])  # Including zero at the start for the initial condition

        return W, t

    def simulate_geometric_brownian_motion(
            self, S0: float, mu: float, sigma: float, T: float, dt: float, n: int
            ) -> Tuple[NDArray, NDArray]:
        """
        Simulate n paths of the Black-Scholes process, each starting at S0.

        Parameters:
        - S0: Initial price
        - mu: Drift coefficient
        - sigma: Volatility
        - T: Time horizon
        - dt: Time step size
        - n: Number of paths to simulate

        Returns:
        - A NumPy array of simulated stock prices, shape (n, num_steps+1)
        - A NumPy array of time steps
        """

        W, t = self.simulate_brownian_motion(T=T, dt=dt, n=n)

        # Calculate paths
        X = (mu - 0.5*sigma**2) * t + sigma * W
        S = S0 * np.exp(X)  # geometric brownian motion paths

        return S, t
    
    def simulate_kou_jump_diffusion(
            self, S0: float, mu: float, sigma: float, lambda_: float, p: float, eta1: float, eta2: float, 
            T: float, dt: float, n: int) -> Tuple[NDArray, NDArray]:
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
        - dt: Time step size
        - n: Number of paths to simulate

        Returns:
        - A NumPy array of simulated stock prices, shape (n, num_steps+1)
        - A NumPy array of time steps
        """
        gbm, t = self.simulate_geometric_brownian_motion(S0=S0, mu=mu, sigma=sigma, T=T, dt=dt, n=n)

        # Jump component
        dv = np.ones((n, len(t)))
        # increments of Poisson process
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


##### debug / test #####
if __name__ == "__main__": 
    import matplotlib.pyplot as plt
    from src.features.data_transformer import Transformer

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
        "mu": 0.12, 
        "sigma": 0.2, 
        "lambda_": 2.0, 
        "p": 0.3, 
        "eta1": 50., 
        "eta2": 25., 
        "T": 10., 
        "dt": 1/250, 
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
