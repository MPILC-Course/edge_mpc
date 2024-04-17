import polars as pl 
import numpy as np 
from tqdm import tqdm 
from datetime import datetime



def kalman_filter(df:pl.DataFrame, h_meas:pl.Series, Qout_meas:pl.Series, A:int=18, Ts:int=1):
    F = np.array([[1,0,0],[Ts/A,1,-Ts/A],[0, 0, 1]])
    C = np.array([[0,1,0],[0, 0, 1]])
    
    xhat_corr_hist = np.zeros((df.shape[0],3))
    
    xhat_corr = np.array([0,0,0])
    xhat_pred = np.array([0,0,0])
    Phat_corr = np.diag([0,0,0])
    Phat_pred = np.diag([0,0,0])
    
    G_w = np.eye(3)
    V = np.diag([0.001,0.001]) # Measurement Noise TODO: Measure on data
    W = np.diag([0.01,1,1e-6]) # Process Noise
    
    for i in tqdm(range(1, df.shape[0])):
        # Kalman Filter
        xhat_pred = F @ xhat_corr  # No input
        Phat_pred = F @ Phat_corr @ F.T + G_w @ W @ G_w.T # A propri covariance
    
        L = Phat_pred @ C.T @ np.linalg.inv(C @ Phat_pred @ C.T + V)
    
        y_k = np.array([h_meas[i], Qout_meas[i]]) # Latest measurement at step k
        xhat_corr = xhat_pred + L @ (y_k - C @ xhat_pred)
        xhat_corr_hist[i] = xhat_corr
        Phat_corr = (np.eye(3) - L @ C) @ Phat_pred
    
    inflow = xhat_corr_hist[:,0]
    outflow = xhat_corr_hist[:,1]
    height = xhat_corr_hist[:,2]
    
    return inflow, outflow, height

def filter_datetime(df, dates_filter, date_format = '%Y-%m-%d %H:%M:%S%z'):
    
    return df.filter(pl.col("time").is_between(datetime.strptime(dates_filter[0], date_format),
                                             datetime.strptime(dates_filter[1], date_format)))
    
if __name__ == '__main__':
    
    precip_df = pl.read_parquet("./data/weather_precip_past1min.par")
    df = pl.read_parquet("./data/static_models/pressure_miso.par")
    inflow_kf, outflow_kf, height_kf = kalman_filter(df, df["level"], df["outflow"])
    df_with_inflow = df.with_columns(inflow = inflow_kf)
    low_df = filter_datetime(df_with_inflow, dates_filter  = ['2023-05-23 00:00:00+00:00', '2023-06-20 23:59:00+00:00'])
    high_df = filter_datetime(df_with_inflow, dates_filter = ['2023-06-27 00:00:00+00:00', '2023-08-03 23:59:00+00:00'])

    df_with_inflow_filtered_1s = pl.concat([low_df, high_df])
    #df_with_inflow_filtered_1s.write_parquet("sym_df_1s_res.parquet")

    df_with_inflow_filtered_1min = df_with_inflow_filtered_1s.group_by_dynamic(index_column="time",
                                                every="1m",
                                                check_sorted=False).agg(pl.col(
                                                    df_with_inflow_filtered_1s.columns[1:]).mean())

    df_with_inflow_filtered_1min_with_rainfall = df_with_inflow_filtered_1min.join(precip_df, on="time", how="inner")

    #df_with_inflow_filtered_1min_with_rainfall.write_parquet("sym_df_1m_res_with_inflow.parquet")
