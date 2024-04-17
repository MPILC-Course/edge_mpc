import numpy as np

class Preprocessor: 

    def _check_negative_values(self, x):
        return 0 if x < 0 else x
    
    def _sqrt_scaler(self, x):
        return np.sqrt(max(x, 0) + 1e-3)
    
    def _sqrt_inverse_transform(self, x):
        return x**2 - 1e-3
    
    def fit_transform(self, df):
        tranformed_df =  df.applymap(self._check_negative_values)
        tranformed_df = tranformed_df.applymap(self._sqrt_scaler)
        return tranformed_df
    
    def inverse_transform(self, df):
        return df.applymap(self._sqrt_inverse_transform) 