import matplotlib.pyplot as plt
import pandas as pd 
import numpy as np 
import warnings
from src.sysidentpy.metrics import mean_squared_error
from src.sysidentpy.model_structure_selection import FROLS
from src.sysidentpy.basis_function._basis_function import Polynomial, Fourier
from src.sysidentpy.residues.residues_correlation import compute_residues_autocorrelation
from src.sysidentpy.residues.residues_correlation import compute_cross_correlation
from src.sysidentpy.utils.display_results import results
from src.sysidentpy.metrics import root_relative_squared_error
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import plotly.graph_objects as go
from plotly_resampler import FigureWidgetResampler
pd.options.mode.chained_assignment = None

warnings.simplefilter(action='ignore', category=FutureWarning)

class ARXParametersIdentification(FROLS):
    def __init__(self, FEATURES, TARGET, dataframe: pd.DataFrame, model_type: str):
        self.FEATURES = FEATURES
        self.TARGET = TARGET
        self.model_type = model_type

    def _slice_data_to_fit(self, x, y):

        if self.model_type == 'miso':
            x = x.values
            y = y.values.reshape(-1,1)

        elif self.model_type == 'siso':
            x = x.values.reshape(-1,1)
            y = y.values.reshape(-1,1)
        return x, y
    


    def identify_arx_parameters(self, x_train, y_train, params_as_dataframe:bool=True, **kwargs):
        
        #create the model
        self.model = FROLS(**kwargs)
        
        
        x_train, y_train = self._slice_data_to_fit(x_train, y_train)

        # Fit the model
        self.fitted_model = self.model.fit(X=x_train, y=y_train)

        # Determine if parameters should be returned as a DataFrame
        params_as_dataframe = kwargs.get('params_as_dataframe', True)
        
        if params_as_dataframe:
            model_parameters = pd.DataFrame(
                results(
                    self.model.final_model,
                    self.model.theta,
                    self.model.err,
                    self.model.n_terms,
                    err_precision=4,
                    dtype="sci",
                ),
                columns=["Regressors", "Parameters", "ERR"]
            )
        else:
            model_parameters = self.model.theta

        return self.fitted_model, model_parameters
    
    def predict(self, x_test:np.array, y_test:np.array, n_steps_ahead:int = 1):

        x_test, y_test = self._slice_data_to_fit(x_test, y_test)

        self.yhat = self.fitted_model.predict(X = x_test, y = y_test, steps_ahead = n_steps_ahead)
        
        rrse = root_relative_squared_error(y_test, self.yhat)
        mse = mean_squared_error(y_test, self.yhat)
        print(f'RRSE: {rrse}')
        print(f'MSE: {mse}')
        
        return self.yhat.flatten()
    

    @staticmethod
    def residuals_analysis(y_test, yhat):
        residuals = y_test - yhat
        fig, axs = plt.subplots(1, 3, figsize=(10, 3))
        # Histogram of residuals
        axs[0].hist(residuals, bins=15, color='skyblue', edgecolor='black')
        axs[0].set_title('Histogram of Residuals')
        axs[0].set_xlabel('Residual Value')
        axs[0].set_ylabel('Frequency')
        axs[0].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Q-Q plot of residuals
        stats.probplot(residuals, dist="norm", plot=axs[1])
        axs[1].set_title('Q-Q Plot of Residuals')
        axs[1].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Scatter plot of residuals
        axs[2].plot(range(0,len(residuals)), residuals, linestyle='none', marker='o', color='red', alpha=0.5)
        axs[2].axhline(y=0, color='blue', linestyle='--')
        axs[2].set_title('Residuals vs. Predicted')
        axs[2].set_xlabel('Predicted Values')
        axs[2].set_ylabel('Residuals')
        axs[2].grid(True, which='both', linestyle='--', linewidth=0.5)

        # Adjust layout for better presentation
        plt.tight_layout()
        plt.close()

        return fig
    
    
    @staticmethod
    def plot_features_and_target(df, features):
        fig = FigureWidgetResampler(go.Figure())
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))

        # Add a trace for each feature
        for feature in features:
            fig.add_trace(
                go.Scattergl(name=f'{feature}', showlegend=True),
                hf_x=df.index,
                hf_y=df[feature]
            )

        fig.update_layout(height=400, template="plotly_dark")
        return fig

