import os
import time
from collections import defaultdict
import numpy as np
import pandas as pd
import importlib
import scipy
import matplotlib.pyplot as plt
import warnings

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.dyn_models.utils import DAEModel
from tops.examples.user_models.user_lib.MyTools.Statistics import EulerMaryama
from tops.examples.user_models.user_lib.MyTools.Functions import (
    calculateSystemTf, makeFFT
)

class ConstPowerLoad(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = self.par = data
        self.n_units = len(data)
        self.bus_idx = np.zeros(self.n_units, dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.zeros(self.n_units, dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def apparent_power_injections(self, x, v):
        s_inj = -(self.par['P'] + 1j * self.par['Q']) / self.sys_par['s_n']
        return self.bus_idx_red['terminal'], s_inj





def run_simulation_batch(model_module_name, t_end=100, theta=0.0075, sigma_p=1.3, sigma_q=0.0,
                         n_simulations=2, time_step=0.02, save_dir='SimulationStorage', max_retries_per_sim=5):

    # Create a unique subfolder name based on input parameters
    subfolder_name = f"theta_{theta}_sigmaP_{sigma_p}_sigmaQ_{sigma_q}_N_{n_simulations}"
    save_dir = os.path.join(save_dir, subfolder_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print(f"Saving results to: {save_dir}")

    # Load model module
    model_data = importlib.import_module(model_module_name)
    importlib.reload(model_data)

    # Precompute transfer function once
    w, mag, w_k2a, mag_k2a, w_req, mag_req, w_reqs, actualReq = calculateSystemTf(model_data)
    # Ensure all sublists are of the same length by padding with NaN
    max_length = max(max(len(w_gen) for w_gen in w), max(len(w_k2a_gen) for w_k2a_gen in w_k2a))

    # Pad each generator's data to the maximum length
    padded_w = [np.append(w_gen, [np.nan] * (max_length - len(w_gen))) for w_gen in w]
    padded_mag = [np.append(mag_gen, [np.nan] * (max_length - len(mag_gen))) for mag_gen in mag]
    padded_w_k2a = [np.append(w_k2a_gen, [np.nan] * (max_length - len(w_k2a_gen))) for w_k2a_gen in w_k2a]
    padded_mag_k2a = [np.append(mag_k2a_gen, [np.nan] * (max_length - len(mag_k2a_gen))) for mag_k2a_gen in mag_k2a]

    # Create the DataFrame
    tf_data = pd.DataFrame({
        'w_Gen1': padded_w[0], 'mag_Gen1': padded_mag[0],
        'w_k2a_Gen1': padded_w_k2a[0], 'mag_k2a_Gen1': padded_mag_k2a[0],
        'w_Gen2': padded_w[1], 'mag_Gen2': padded_mag[1],
        'w_k2a_Gen2': padded_w_k2a[1], 'mag_k2a_Gen2': padded_mag_k2a[1],
        'w_Gen3': padded_w[2], 'mag_Gen3': padded_mag[2],
        'w_k2a_Gen3': padded_w_k2a[2], 'mag_k2a_Gen3': padded_mag_k2a[2],
        'w_Gen4': padded_w[3], 'mag_Gen4': padded_mag[3],
        'w_k2a_Gen4': padded_w_k2a[3], 'mag_k2a_Gen4': padded_mag_k2a[3],
    })

    # Save the DataFrame to a CSV file
    tf_data.to_csv(os.path.join(save_dir, "system_transfer_function.csv"), index=False)

    successful_sims = 0
    attempt = 0

    while successful_sims < n_simulations:
        attempt += 1
        print(f"\n--- Simulation Attempt {attempt} (Successes: {successful_sims}/{n_simulations}) ---")
        
        try:
            model = model_data.load()
            model['loads'] = {
                #'Load': [model['loads'][ix] for ix in [0, 2]],
                'ConstPowerLoad': [model['loads'][ix] for ix in [0, 1, 2]]
            }

            user_mdl_lib = type('', (), {'loads': type('', (), {'ConstPowerLoad': ConstPowerLoad})})
            ps = dps.PowerSystemModel(model=model, user_mdl_lib=user_mdl_lib)

            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                ps.power_flow()  # Explicitly run this if not already in init_dyn_sim

            ps.init_dyn_sim()

            # Proceed with simulation
            idx1 = 0
            x_0 = ps.x_0.copy()
            p_0 = ps.loads['ConstPowerLoad'].par['P'][0]
            mu_p = p_0

            sol = dps_sol.ModifiedEulerDAE(
                ps.state_derivatives, ps.solve_algebraic, 0, x_0, ps.v0, t_end, max_step=time_step)

            res = defaultdict(list)
            vec1 = np.arange(0, 300000, time_step)
            line_mdl = ps.lines['Line']

            def p_bus_7(x, v):
                return -ps.s_n * (line_mdl.p_to(x, v)[1] + line_mdl.p_from(x, v)[2] + line_mdl.p_from(x, v)[3])
            
            def p_bus_9(x, v):
                return -ps.s_n * (line_mdl.p_to(x, v)[4] + line_mdl.p_to(x, v)[5] + line_mdl.p_from(x, v)[6])
            
            t = 0
            while t < t_end:
                if t > vec1[idx1]:
                    ps.loads['ConstPowerLoad'].par['P'][0] = EulerMaryama(
                        theta, mu_p, sigma_p, time_step, ps.loads['ConstPowerLoad'].par['P'][0])
                    idx1 += 1

                sol.step()
                x, v, t = sol.y, sol.v, sol.t
                res['t'].append(t)
                res['freq'].append(((ps.gen['GEN'].speed(x, v).copy() * 50)) + 50)
                res['p_bus_7'].append(p_bus_7(x, v).copy())
                res['p_bus_9'].append(p_bus_9(x, v).copy())

            avg_freq = [np.mean(freq) for freq in res['freq']]
            res['avg_freq'] = avg_freq

            df = pd.DataFrame({
                'time': res['t'],
                'freq': res['freq'],
                'avg_freq': avg_freq,
                'p_bus_7': np.abs(res['p_bus_7']),
                'p_bus_9': np.abs(res['p_bus_9'])
            })
            df.to_csv(os.path.join(save_dir, f"simulation_{successful_sims + 1}_results.csv"), index=False)

            p_combined = (np.abs(res['p_bus_7']) - np.mean(np.abs(res['p_bus_7']))) + \
                        (np.abs(res['p_bus_9']) - np.mean(np.abs(res['p_bus_9'])))
            p_fft_freq_rad, p_fft_mag = makeFFT(p_combined / 33.4, time_step)

            df_fft = pd.DataFrame({'frequency_rad': p_fft_freq_rad, 'magnitude': p_fft_mag})
            df_fft.to_csv(os.path.join(save_dir, f"simulation_{successful_sims + 1}_fft.csv"), index=False)

            successful_sims += 1

        except RuntimeWarning as w:
            print(f"⚠️ Power flow or solver failed: {w}. Retrying...")
            continue
        except Exception as e:
            print(f"❌ Unexpected error during simulation: {e}. Retrying...")
            continue

    print(f"\n✅ All {n_simulations} simulations completed. Results saved to '{save_dir}'.")

# run_simulation_batch('tops.examples.user_models.user_lib.ProsjektoppgaveMaster.k2aTuning',
#                      t_end=2000, theta=0.013, sigma_p=1.34, sigma_q=0.0, n_simulations=50, time_step=0.02,
#                      save_dir='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/src/tops/examples/dyn_models/ConstPowerStochasticLoadModels/SimulationStorage')

# run_simulation_batch(
#     model_module_name='tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a',
#     t_end=2000,
#     theta=0.015,
#     sigma_p=28,
#     sigma_q=0,
#     n_simulations=100,
#     time_step=0.02,
#     save_dir='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/src/tops/examples/dyn_models/ConstPowerStochasticLoadModels/SimulationStorage')


# run_simulation_batch(
#     model_module_name='tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a',
#     t_end=2000,
#     theta=0.015,
#     sigma_p=30,
#     sigma_q=0,
#     n_simulations=100,
#     time_step=0.02,
#     save_dir='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/src/tops/examples/dyn_models/ConstPowerStochasticLoadModels/SimulationStorage')

# run_simulation_batch(
#     model_module_name='tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a',
#     t_end=2000,
#     theta=0.015,
#     sigma_p=28.5,
#     sigma_q=0,
#     n_simulations=100,
#     time_step=0.02,
#     save_dir='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/src/tops/examples/dyn_models/ConstPowerStochasticLoadModels/SimulationStorage')

# run_simulation_batch(
#     model_module_name='tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a',
#     t_end=2000,
#     theta=0.0125,
#     sigma_p=27.5,
#     sigma_q=0,
#     n_simulations=100,
#     time_step=0.02,
#     save_dir='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/src/tops/examples/dyn_models/ConstPowerStochasticLoadModels/SimulationStorage')

# run_simulation_batch(
#     model_module_name='tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a',
#     t_end=2000,
#     theta=0.0125,
#     sigma_p=30,
#     sigma_q=0,
#     n_simulations=100,
#     time_step=0.02,
#     save_dir='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/src/tops/examples/dyn_models/ConstPowerStochasticLoadModels/SimulationStorage')


# run_simulation_batch(
#     model_module_name='tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a',
#     t_end=2000,
#     theta=0.01,
#     sigma_p=20,
#     sigma_q=0,
#     n_simulations=100,
#     time_step=0.02,
#     save_dir='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/src/tops/examples/dyn_models/ConstPowerStochasticLoadModels/SimulationStorage')


def plot_frequency_results(file_path):
    all_data = []  # List to store all frequency data
    time_data = None  # To store the time column (assumes all files have the same time steps)
    plt.figure()
    # List all files in the directory
    for file_name in os.listdir(file_path):
        # Check if the file matches the pattern "simulation_<number>_results.csv"
        if file_name.startswith("simulation_") and file_name.endswith("_results.csv"):
            # Construct the full file path
            full_path = os.path.join(file_path, file_name)
            
            # Read the CSV file
            data = pd.read_csv(full_path)
            
            # Check if the 'avg_freq' column exists in the data
            if 'avg_freq' in data.columns:
                # Append the frequency data to the list
                all_data.append(data['avg_freq'].values)
                # Store the time column (assumes all files have the same time steps)
                if time_data is None:
                    time_data = data['time'].values
                
                # Plot the individual frequency curve
                plt.plot(data['time'], data['avg_freq'], alpha=0.5)  # Add transparency for clarity
            else:
                print(f"Skipping {file_name}: 'avg_freq' column not found.")

    # Calculate the mean curve
    if all_data:
        mean_curve = np.mean(all_data, axis=0)  # Take the mean across all simulations
        # Plot the mean curve
        plt.plot(time_data, mean_curve, label='Mean Curve', color='salmon', linewidth=2.5)


    # Add labels and legend to the frequency plot
    plt.xlabel('Time')
    plt.ylabel('Average Frequency')
    plt.title('Frequency Data from Simulations')
    plt.legend()
    plt.show()


# Plot the transfer functions
    transfer_function_file = os.path.join(file_path, "system_transfer_function.csv")
    if os.path.exists(transfer_function_file):
        tf_data = pd.read_csv(transfer_function_file)
        plt.figure()  # Create a new figure for the transfer function plot
        for gen_idx in range(1, 5):  # Assuming 4 generators
            w_col = f"w_Gen{gen_idx}"
            mag_col = f"mag_Gen{gen_idx}"
            if w_col in tf_data.columns and mag_col in tf_data.columns:
                plt.plot(tf_data[w_col], tf_data[mag_col], label=f"Generator {gen_idx}")

        # Customize the transfer function plot
        plt.xscale('log')  # Logarithmic scale for frequency
        plt.xlabel('Frequency (rad/s)')
        plt.ylabel('Magnitude')
        plt.title('Transfer Function')
        plt.legend()
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.show()





# Example usage
plot_frequency_results('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/src/tops/examples/dyn_models/ConstPowerStochasticLoadModels/SimulationStorage/theta_0.013_sigmaP_1.34_sigmaQ_0.0_N_50')

