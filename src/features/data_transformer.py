import numpy as np
import pandas as pd
from numpy.typing import NDArray
from esig import tosig as ts

# encoder transforms data frame time series
class Transformer:
    def __init__(self, paths_df: pd.DataFrame = None, paths: NDArray = None, time: NDArray = None):
        self.paths_df = paths_df
        self.paths = paths
        self.time = time

    def calculate_returns_df(self, shift: int, logFlag: bool) -> pd.DataFrame:
        """
        Calculate shift-day returns for a DataFrame of prices. Returns are spaced by shift days,
        not rolling, resulting in a DataFrame of length 1/n of the original.

        Parameters:
        - shift: int, the number of days for calculating returns.

        Returns:
        - A pandas DataFrame with shift-day returns, having the same column structure and a reduced index.
        """
        # Forward fill to handle any missing values
        filled_df = self.paths_df.ffill()
        # only keep every shift observation
        reduced_df = filled_df.iloc[::shift]
        if logFlag:
            # Calculate returns using the logarithmic method
            returns_df = np.log(reduced_df / reduced_df.shift(1))
        else:
            # Calculate simple returns
            returns_df = reduced_df.pct_change()
        
        return returns_df

    def calculate_daily_rolling_returns_df(self, shift: int):
        return self.paths_df.pct_change(periods = shift)
    
    def calculate_daily_rolling_log_returns_df(self, shift: int):
        return np.log(self.paths_df / self.paths_df.shift(shift))
    
    def stream2logsig_wrapper(self, path: NDArray, times: NDArray, depth: int):
        # Helper function to call stream2logsig function with path and time and avoid loops
        return ts.stream2logsig(np.column_stack((times, path)), depth)
    
    def stream2sig_wrapper(self, path: NDArray, times: NDArray, depth: int):
        # Helper function to call stream2sig function with path and time and avoid loops
        return ts.stream2sig(np.column_stack((times, path)), depth)

    def calculate_signature(self, depth: int, type: str = "log") -> NDArray:
        if self.time is None or self.paths is None:
            raise ValueError('Both numpy arrays `time` and `paths` must be provided for signature calculation.')
        if type not in ("log", "normal"):
            raise ValueError(f'{type = } is not a valid argument for signature calculation. Must be "normal" or "log".')

        # Apply wrapper to all rows of self.paths along with self.time
        # Each row of the output will correspond to the (log-)signature of one path
        if type == "log":
            return np.apply_along_axis(self.stream2logsig_wrapper, axis=1, arr=self.paths, times=self.time, depth=depth)
        else: 
            return np.apply_along_axis(self.stream2sig_wrapper, axis=1, arr=self.paths, times=self.time, depth=depth)


##### test / debug #####
if __name__ == "__main__":
    pass