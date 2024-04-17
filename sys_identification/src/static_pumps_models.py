import pandas as pd
import pwlf
import numpy as np 
import matplotlib.pyplot as plt


def filter_data(df, pump_number, feature, bin_size, pump_selection = False):
    # Prepare column names based on the selected pump number
    if pump_selection == True:
        if pump_number == 1:
            df[(df['pump1_speed'] > 0) & (df['pump3_speed'] == 0) & (df['pump4_speed'] == 0)]
        elif pump_number == 3:
            df[(df['pump1_speed'] == 0) & (df['pump3_speed'] >= 0) & (df['pump4_speed'] == 0)]
        elif pump_number == 4:
            df[(df['pump1_speed'] == 0) & (df['pump3_speed'] == 0) & (df['pump3_speed'] > 0)]
            
        pump_speed_col = f'pump{pump_number}_speed'
    
    # Copy the DataFrame to avoid modifying the original data
    xdf = df.copy()
    max_speed = xdf[pump_speed_col].max()
    bins = range(0, int(max_speed) + bin_size, bin_size)
    
    # Bin the pump speeds
    xdf['speed_bin'] = pd.cut(xdf[pump_speed_col], bins, right=False)
    
    # Group by the new 'speed_bin' column and calculate mean values
    xdf = xdf.groupby('speed_bin')[[feature, pump_speed_col]].mean().reset_index()
    
    # Calculate the midpoint of each speed bin
    xdf['speed_mid'] = xdf['speed_bin'].apply(lambda x: (x.left + x.right) / 2)
    
    return xdf




outflow_df  = pd.read_parquet("./data/static_models/outflow_miso.par")

pump1_power_df = pd.read_parquet("./data/static_models/pump1_power_siso.par")
pump4_power_df = pd.read_parquet("./data/static_models/pump4_power_siso.par")
pump3_power_df = pd.read_parquet("./data/static_models/pump3_power_siso.par")



outflow_df_filtered = filter_data(df = outflow_df,
                                      pump_number = 1,
                                      feature = "outflow",
                                      bin_size = 20, 
                                      pump_selection=False)



import pwlf
breakpoints = [np.min(outflow_df_filtered["pump1_speed"]), 750, np.max(outflow_df_filtered["pump1_speed"])]

outflow_df_filtered["pump1_speed"] = outflow_df_filtered["pump1_speed"].shift(1)

outflow_df_filtered = outflow_df_filtered.dropna()

my_pwlf = pwlf.PiecewiseLinFit(outflow_df_filtered["pump1_speed"], outflow_df_filtered["outflow"])
my_pwlf.fit_with_breaks(breakpoints)
y_hat = my_pwlf.predict(outflow_df_filtered["pump1_speed"])

plt.scatter(outflow_df_filtered["speed_mid"], outflow_df_filtered["outflow"])
plt.scatter(outflow_df_filtered["pump1_speed"], my_pwlf.predict(outflow_df_filtered["speed_mid"]))