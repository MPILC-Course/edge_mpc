import casadi as ca
import numpy as np
import pandas as pd 
from setup_OptiProblem import setup_OptiProblem
import time
import os

df = pd.read_parquet("/home/pi/mpc/edge_mpc/data/sym_data/sym_df_5s_res_withPower.parquet")
# df = df["2023-05-27 15:00:00":"2023-06-27 20:00:00"]
timestamps = df.index.tolist()

zs = 2
N = 24
Ts = 5
mpc_controller = setup_OptiProblem(N,zs,Ts)

# =================== History Initialization ===================
# ==== Last element[-1]: latest; first[0]: oldest
time_hist = timestamps[0:zs]
timings =  [0]*zs
h_hist = df["level"][0:zs].values
w_hist = np.vstack([df["pump1_speed"][0:zs].values,df["pump3_speed"][0:zs].values,df["pump4_speed"][0:zs].values]).reshape(3,-1)
Qout_hist = df["outflow"][0:zs].values
E_hist = np.vstack([df["pump1_power_est"][0:zs].values,df["pump3_power_est"][0:zs].values,df["pump4_power_est"][0:zs].values]).reshape(3,-1)
P_hist = df["pressure"][0:zs].values
effi_hist = [0,0]
Qin_hist = df["inflow"][0:zs].values


# construct pump trigger signal based on external schedule
trigger_k = [1,0,0]
p = 0

Qin_k = df["inflow"].values

# building some trajectory for height reference
h_ref_k = np.ones(len(df))*120
h_ref_hist = h_ref_k[0:zs] # history of reference for plotting, href_k cant be plotted because time horizon 


# =================== Measurement Updates and appending ===================
K = len(df)-150
for k in range(K): #len(df)-150
    
    # Actual Solving

    sol_start = time.time() # Timing
    sol_w,sol_Qout,sol_h,sol_E,sol_P,sol_effi = mpc_controller(Qin_k[k+zs:k+zs+N+zs],Qout_hist[-zs:],h_hist[-zs:],w_hist[:,-zs:],E_hist[:,-zs:],P_hist[-zs:],trigger_k,h_ref_k[k:N+k+zs])
    sol_end = time.time()

    # Logging   
    time_hist = np.hstack([time_hist, timestamps[k+zs]])
    timings.append(sol_end-sol_start)

    h_meas_k = sol_h[zs].full().reshape(1)
    h_hist = np.hstack([h_hist, h_meas_k])

    w_meas_k = sol_w[:,zs].full()
    w_hist = np.hstack([w_hist, w_meas_k])

    Qout_meas_k = sol_Qout[zs].full().reshape(1)
    Qout_hist = np.hstack([Qout_hist, Qout_meas_k])  

    E_meas_k = sol_E[:,zs].full()
    E_hist = np.hstack([E_hist, E_meas_k])   

    P_meas_k = sol_P[zs].full().reshape(-1)
    P_hist = np.hstack([P_hist, P_meas_k]) 

    effi_k = sol_effi[zs].full().reshape(-1)
    effi_hist = np.hstack([effi_hist, effi_k])   

    h_ref_hist = np.hstack([h_ref_hist, h_ref_k[k+zs]])   
    
    Qin_hist = np.hstack([Qin_hist, Qin_k[k+zs]]) 

    # Alternating Pump Schedule, 4000 samples on (6h)

    if ((k+100)%4000 == 0):
        p = p+1
        if (p>2): 
            p = 0
        trigger_k[p] = 1

    if (k%4000 == 0):
        if (p==0): 
            trigger_k[2] = 0
        else:
            trigger_k[p-1] = 0


    # Save subresults every 5000 steps
    if (k%5000 == 0):
        results = pd.DataFrame(list(zip(w_hist[0,:].reshape(-1), w_hist[1,:].reshape(-1), w_hist[2,:].reshape(-1), Qout_hist,P_hist,h_hist,Qin_hist,E_hist[0,:].reshape(-1), E_hist[1,:].reshape(-1), E_hist[2,:].reshape(-1))), index= time_hist,columns=df.columns)
        results["t_wall"] = timings
        results.to_parquet(f'/home/pi/mpc/edge_mpc/data/results/mpc_subresults_{k}.parquet')
        if os.path.exists(f'/home/pi/mpc/edge_mpc/data/results/mpc_subresults_{k-5000}.parquet'):
            os.remove(f'/home/pi/mpc/edge_mpc/data/results/mpc_subresults_{k-5000}.parquet')
        
    print(f"iteration : {k+1}/{K},    t_wall: {timings[k+zs]}")
    
           

results = pd.DataFrame(list(zip(w_hist[0:,:].reshape(-1), w_hist[1:,:].reshape(-1), w_hist[2:,:].reshape(-1), Qout_hist,P_hist,h_hist,Qin_hist,E_hist[0:,:].reshape(-1), E_hist[1:,:].reshape(-1), E_hist[2:,:].reshape(-1))), index= time_hist,columns=df.columns)
results["t_wall"] = timings
results.to_parquet('/home/pi/mpc/edge_mpc/data/results/mpc_results.parquet')



