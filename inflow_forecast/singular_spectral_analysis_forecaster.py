import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

class SSA(object):

    __supported_types = (pd.Series, np.ndarray, list)

    def __init__(self, tseries, L):
        """
        Decomposes the given time series with a singular-spectrum analysis. Assumes the values of the time series are
        recorded at equal intervals.

        Parameters
        ----------
        tseries : The original time series, in the form of a Pandas Series, NumPy array or list.
        L : The window length. Must be an integer 2 <= L <= N/2, where N is the length of the time series.
        save_mem : Conserve memory by not retaining the elementary matrices. Recommended for long time series with
            thousands of values. Defaults to True.

        Note: Even if an NumPy array or list is used for the initial time series, all time series returned will be
        in the form of a Pandas Series or DataFrame object.
        """

        # Tedious type-checking for the initial time series
        if not isinstance(tseries, self.__supported_types):
            raise TypeError("Unsupported time series object. Try Pandas Series, NumPy array or list.")

        # Checks to save us from ourselves
        self.N = len(tseries)
        if not 2 <= L <= self.N/2:
            raise ValueError("The window length must be in the interval [2, N/2].")

        self.L = L
        self.orig_TS = pd.Series(tseries)
        self.K = self.N - self.L + 1

        # Embed the time series in a trajectory matrix
        self.X = np.column_stack([self.orig_TS.values[i:i+L] for i in range(0,self.K)])

        # Decompose the trajectory matrix
        self.d = np.linalg.matrix_rank(self.X)
        self.U, self.Sigma, VT = np.linalg.svd(self.X)
        self.V = VT.T
        self.Xi = np.array([ self.Sigma[i]*np.outer(self.U[:,i], self.V[:,i]) for i in range(self.d)]) 
        if not np.allclose(self.X, self.Xi.sum(axis=0), atol=1e-10): #error tolerance is very small (order or 10^-10)
            print("WARNING: The sum of X's elementary matrices is not equal to X!")
            
    def X_to_TS(self, Xi):
        """
        Performs (anti-) diagonal averaging on given elementary matrix, Xi, and returns a time series.
        """
        Xrev = Xi[::-1]
        return np.array([Xrev.diagonal(i).mean() for i in range(-Xi.shape[0]+1, Xi.shape[1])])
    
    def get_contributions(self):
        '''Calculate the relative contribution of each of the singular values'''
        lambdas = np.power(self.Sigma,2)
        frob_norm = np.linalg.norm(self.X)
        self.ret = pd.DataFrame(lambdas/(frob_norm**2), columns=['Contribution']) # frobenius norm
        self.ret['Contribution'] = self.ret["Contribution"].round(4)
        self.ret= self.ret[self.ret["Contribution"]>0]
        return self.ret

    def forecast_recurrent(self, principal_components_indexes: list, n_steps = int):
        pricipal_components_forecast = []
        s_contributions = self.get_contributions()
        r = len(s_contributions[s_contributions>0])
        self.r_characteristic = round((self.Sigma[:r]**2).sum()/(self.Sigma**2).sum(),4)
        forecast_orthonormal_base = {i:self.U[:,i] for i in range(r)}
        X_container = np.zeros(self.X.shape)
        verticality_coefficient = 0
        R = np.zeros(forecast_orthonormal_base[0].shape)[:-1]
        for Pi in forecast_orthonormal_base.values():
            X_container += np.dot(((Pi*Pi.T).reshape((1,self.L))),self.X)
            pi = np.ravel(Pi)[-1]
            verticality_coefficient += pi**2
            R += pi*Pi[:-1]  
        R = (R/(1-verticality_coefficient))
        ts_selected_components = self.X_to_TS(self.Xi[principal_components_indexes].sum(axis=0))
        self.R = R[::-1]
        forecast_array = np.zeros(n_steps)
        ts_selected_components_list = list(ts_selected_components)
        for i in range(self.N, self.N+n_steps):
            s = 0
            for j in range(self.L-1):
                s += (R[j]*ts_selected_components_list[i-j-1])
                pricipal_components_forecast.append((R[j]*ts_selected_components_list[i-j-1]))
            ts_selected_components_list.append(s)
        forecast_array = ts_selected_components_list[self.N:]
        pricipal_components_forecast = np.array(pricipal_components_forecast).reshape(n_steps, self.L-1)
        pricipal_components_forecast = pricipal_components_forecast[:, principal_components_indexes]
        cols = ["F{}".format(i) for i in range(len(principal_components_indexes))]
        pricipal_components_forecast_df = pd.DataFrame(pricipal_components_forecast, columns=cols, index=np.arange(self.N, self.N+n_steps))
        return  pricipal_components_forecast_df, forecast_array
    
    def plot_ntop_components(self, ntop_components):
        cols = plt.get_cmap('tab10').colors
        plt.rcParams['axes.prop_cycle'] = cycler(color=cols)
        t = np.arange(0, self.N)
        fig = plt.subplot()
        color_cycle = cycler(color=plt.get_cmap('tab20').colors)
        fig.axes.set_prop_cycle(color_cycle)
        # Convert elementary matrices straight to a time series
        for i in range(0,ntop_components):
            Fi = self.X_to_TS(self.Xi[i])
            fig.axes.plot(t, Fi, lw=1)#lw stands for linewidth

        fig.set_xlabel("$t$")
        fig.set_ylabel(r"$\tilde{F}_i(t)$")
        legend = [r"$\tilde{F}_{%s}$" %i for i in range(ntop_components)] + ["$F$"]
        fig.set_title("The First n Components of the Time Series")
        fig.legend(legend, loc=(1.05,0.1))
        fig.grid();
            
        
    def components_to_df(self, n=0):
        """
        Returns all the time series components in a single Pandas DataFrame object.
        """
        if n > 0:
            n = min(n, self.d)
        else:
            n = self.d
        
        # Create list of columns - call them F0, F1, F2, ...
        cols = ["F{}".format(i) for i in range(n)]
        
        # Calculate the time series components
        TS_comps = np.zeros((self.N, n))
        for i in range(n):
            Fi = self.X_to_TS(self.Xi[i])
            TS_comps[:, i] = Fi
        
        return pd.DataFrame(TS_comps, columns=cols, index=np.arange(self.N, self.N+len(Fi)))
 


