import numpy as np
import pandas as pd

# encoder transforms data frame time series
class Transformer:
    def __init__(self, paths_df: pd.DataFrame):
        self.paths_df = paths_df

    def calculate_returns(self, shift: int, logFlag: bool) -> pd.DataFrame:
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

    def calculate_daily_rolling_returns(self, shift: int):
        return self.paths_df.pct_change(periods = shift)
    
    def calculate_daily_rolling_log_returns(self, shift: int):
        return np.log(self.paths_df / self.paths_df.shift(shift))
    
    def calculate_log_signature(self):
        pass
    
    def calculate_signature(self):
        pass

##### test / debug #####
if __name__ == "__main__":
    pass