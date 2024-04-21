import numpy as np
from numpy.typing import NDArray

##### remove later ############
import matplotlib.pyplot as plt
################################

class DataLoader:
    def __init__(self, params: dict[str, float | int]):
        self.params = params
        self.method_functions = {
            "Brownian_Motion": self.simulate_brownian_motion,
            "GBM": self.simulate_geometric_brownian_motion,
            "Kou_Jump_Diffusion": self.simulate_kou_jump_diffusion
        }

    def create_dataset(self, method: str) -> tuple[NDArray, NDArray]:
        if method in self.method_functions:
            return self.method_functions[method](**self.params)
        else:
            raise ValueError(f"Data creation method '{method}' not implemented.")
        
    def simulate_brownian_motion(self, T: float, dt: float, n: int) -> tuple[NDArray, NDArray]:
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
        t = np.linspace(0, T, num_steps + 1)
        dW = np.random.normal(size=(n, num_steps))  # scaled increments
        W = np.cumsum(np.sqrt(dt)*dW, axis=1)  # cumulative sum to generate paths
        W = np.hstack([np.zeros((n, 1)), W])  # Including zero at the start for the initial condition

        return W, t

    def simulate_geometric_brownian_motion(
            self, S0: float, mu: float, sigma: float, T: float, dt: float, n: int
            ) -> tuple[NDArray, NDArray]:
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
            T: float, dt: float, n: int) -> tuple[NDArray, NDArray]:
        """
        Simulate n paths of the Kou jump-diffusion process, each starting at S0.

        Parameters:
        - S0: Initial price
        - mu: Drift coefficient
        - sigma: Volatility
        - lambda_: Jump intensity
        - p: Probability of positive jump
        - eta1: Rate of positive exponential distribution
        - eta2: Rate of negative exponential distribution
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
        N = np.random.poisson(lambda_*dt, size=(n, len(t)-1))

        # TODO: make more efficient and check logic again
        for i in range(n):
            for j in range(1, len(t)):
                # iterate through number of jumps
                vi = np.ones(N[i, j-1])
                for k in range(N[i, j-1]):
                    gamma = np.random.exponential(1/eta1) if np.random.rand() < p else -np.random.exponential(1/eta2)
                    vi[k] = np.exp(gamma)
                dv[i, j] = np.prod(vi)

        # Calculate the paths with jumps
        S = gbm * np.cumprod(dv, axis=1) 
        return S, t

##### debug / test #####
kou_params = {
    "S0": 1., 
    "mu": 0.1, 
    "sigma": 0.16, 
    "lambda_": 1.0, 
    "p": 0.5, 
    "eta1": 10., 
    "eta2": 5., 
    "T": 1., 
    "dt": 0.001, 
    "n": 100
}
gbm_params = {
    "S0": 1., 
    "mu": 0.05,
    "sigma": 0.2, 
    "T": 10., 
    "dt": 0.001, 
    "n": 1000
}
brownian_motion_params = {
    "T": 10., 
    "dt": 0.001, 
    "n": 1000
}
loader = DataLoader(params=kou_params)
prices, times = loader.create_dataset("Kou_Jump_Diffusion")

# Plotting the paths
plt.figure(figsize=(18, 10))
for i in range(prices.shape[0]):
    plt.plot(times, prices[i], linewidth=0.1, alpha=0.4, color='blue')  # Reduced linewidth and added transparency

plt.title('Simulated Paths')
plt.xlabel('Time (Years)')
plt.ylabel('Price')
plt.grid(True)
plt.show()