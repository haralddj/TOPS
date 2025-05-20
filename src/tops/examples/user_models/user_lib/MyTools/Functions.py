import pandas as pd
import matplotlib.pyplot as plt
import tops.dynamic as dps
from tops.examples.interfaces.results_events import ResultKeeper, Events
from tops.simulator import Simulator
import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
#import tops.ps_models.k2a as model_data
import numpy as np
from collections import defaultdict
import time
import sys
from sympy.abc import s
import sympy as sp
from scipy.signal import TransferFunction, bode, welch, periodogram
from scipy.optimize import curve_fit
import os
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf


def runDynSimulation(model, t_end, timestep, H, Tf, Tr, Tg, Tw, R, r):
    #model_data = __import__(model)
    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}
    #model['loads'] = {'ConstantPowerLoad': model['loads']}

    for gen in model['generators']['GEN'][1:]:
        gen[6] = H
        #gen[7] = D
    #Set the time constants of the governors
    for gov in model['gov']['HYGOV'][1:]:
        gov[2]=R
        gov[3]=r
        gov[4]=Tf
        gov[5]=Tr
        gov[6]=Tg
        gov[8]=Tw

    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    ps.ode_fun(0, ps.x0)
    sim = Simulator(ps, dt=timestep, t_end=t_end)
    res_keeper = ResultKeeper(sim)
    events = Events(sim, [
        (17.2, ('dynload', 'g_setp', 1.09, 0)),
        (17.2, ('dynload', 'b_setp', 0.22, 0)),
    ])

    sim.interface_functions['ResultKeeper'] = res_keeper.update
    sim.interface_functions['Events'] = events.update
    sim.main_loop()

    print('Done')
    df = res_keeper.get_dataframe()
    self = res_keeper
    index = pd.MultiIndex.from_tuples([tuple(row) for row in sim.ps.state_desc], names=['Model', 'state'])
    df = pd.DataFrame(columns=index, data=self.x, index=self.t)
    speed_columns = [col for col in df.columns if col[1] == 'speed']

    time, freq=GetFreqData('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous_(1)/Outage-anonymous.xlsx', 'FI south:Frequency')
    freq=freq[0:2000]
    time=time[0:2000]
    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/Taajuusdata2024-05-05.csv')
    freq1=data['Value'][156800:158100]
    time1=[]
    for i in range(len(freq1)):
        time1.append(i/10)

    # Calculate the average speed
    df['average_speed'] = df[speed_columns].mean(axis=1)
    df['avg_freq'] = (df['average_speed'] * 50) + 50.04
    plt.figure(figsize=(8, 6))
    plt.plot(df.index, df['avg_freq'], label='Simulated Frequency')
    plt.plot(time1, freq1, label='Measured Frequency1')
    plt.plot(time, freq, label='Measured Frequency2')
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.xlim(0, 60)
    plt.title('Frequency vs Time')
    plt.legend()
    plt.show()

    time_freq_list = [df.index.tolist(), df['avg_freq'].tolist()]
    return (time_freq_list[0], time_freq_list[1])






def GetFreqData(filepath, columnname):
    # Step 1: Read the Excel file
    data = pd.read_excel(filepath)

    # Step 2: Extract the relevant data
    frequency_data = data[columnname]
    freqlist = frequency_data.tolist()

    # Step 3: Create the time axis
    sampling_rate = 1500 / 60  # 1500 samples per minute
    time_values = [i / sampling_rate for i in range(len(frequency_data))]

    # Step 4: Plot the data
    '''plt.figure(figsize=(10, 6))
    plt.plot(time_values, frequency_data)
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Frequency vs Time')
    plt.grid(True)
    plt.show()'''
    return time_values, freqlist


def GetFreqDataCSV(filepath, columnname):
    # Step 1: Read the Excel file
    data = pd.read_csv(filepath)

    # Step 2: Extract the relevant data
    frequency_data = data[columnname]
    freqlist = frequency_data.tolist()
    time_values=[i/10 for i in range(len(frequency_data))]

    return time_values, freqlist


#For k2a
#runDynSimulation(model_data, 60, 5e-3, 3 , 0.1, 7, 0.6, 3.7, 0.14, 1.8) #, 20, 10, 0.3, 0.9, 0.05, 0.07, 0.04)

def calculateSystemInertia(model):
    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # Calculate the total inertia constant of the system
    total_inertiatimesRating = 0
    totalRating=0
    for gen in model['generators']['GEN'][1:]:
        total_inertiatimesRating += gen[6]*gen[2]  # Assuming the inertia constant is at index 6
        totalRating+=gen[2]
    SystemInertia=total_inertiatimesRating/totalRating
    print(SystemInertia)
    return SystemInertia

'''def calculateSystemLoadFreqDependence(model):   #Må spørre Sigurd om denne
    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # Calculate the total damping constant of the system
    total_dependenceTimesRating = 0
    totalRating=0
    for gen in model['generators']:
        K_d=(2/40)*gen[7]/gen[2]            #Antar 40 poler(?), blir 0 uansett når D=0

        total_dampingtimesRating += gen[7]*gen[2]  # Assuming the damping constant is at index 7
        totalRating+=gen[2]
    SystemDamping=total_dampingtimesRating/totalRating
    print(SystemDamping)
    return SystemDamping'''

def getSystemNominalPower(model):
    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # Calculate the total nominal power of the system
    totalRating=0
    for gen in model['generators']:
        totalRating+=gen[2]
    print(totalRating)
    return totalRating


def upInertia(model, delta):
    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # Increase inertia constant of all generators by delta
    for gen in model['generators']:
        ps.gen[1][6] += delta  # Assuming the inertia constant is at index 6
    return model
def downInertia(model, delta):
    # Power system model
    ps = dps.PowerSystemModel(model=model)
    # Decrease inertia constant of all generators by delta
    for gen in model['generators']:
        ps.gen[1][6] -= delta  # Assuming the inertia constant is at index 6
    return model

def create_Gp(T1, T2, At, Tr, r, Tf, Tg, R, H, Kd,Dg,Tdel):
    Gt = At * (1 - s * T1) / (1 + s * T2)
    Gc = (1 + s * Tr) / (r * (1 + s * Tf) * Tr * s + R * (1 + s * Tr))
    Gs = (1 / (1 + s * Tg))  #SGate Servomotor
    Gj = 1 / (2 * H * s + Kd)
    Gp = (Gt * Gc * Gs)+Dg
    #Gp=((1+Tr*s)/R)/((Tf*Tr*Tg/R)*s**3+((Tf+Tr)*Tg/R)*s**2+((Tg+Tr*(R+r))/R)*s+1)*Gt+Dg
    #Gp=((1/(1+ s * Tf))*(1/Tg)*(1/s))/(1+(1/1+s*Tf)*(1/(Tg*s))*((r*s*Tr)/(1+s*Tr)+R))*Gt
    #print("Gt:", Gt, "Gc:", Gc, "Gs:", Gs, "Gj:", Gj, "Gp:", Gp)
    #sp.pretty_print(sp.simplify(Gp))

    S = 1 / (1 + Gj * Gp)
    G0 = -Gj * S
    return Gp, Gj

def checkGetTf(model):
    ps = dps.PowerSystemModel(model=model)
    Gp_list = []  # List to store Gp for all generators

    # Loop through all generators in the model
    for gen_idx, gen in enumerate(model['generators']['GEN'][1:], start=1):  # Skip the first entry (header)
        Tw = model['gov']['HYGOV'][gen_idx][8]
        q_nl = model['gov']['HYGOV'][gen_idx][9]
        A_t = model['gov']['HYGOV'][gen_idx][7]
        Tr = model['gov']['HYGOV'][gen_idx][5]
        r = model['gov']['HYGOV'][gen_idx][3]
        Tf = model['gov']['HYGOV'][gen_idx][4]
        Tg = model['gov']['HYGOV'][gen_idx][6]
        R = model['gov']['HYGOV'][gen_idx][2]
        H = gen[6]
        Dt = model['gov']['HYGOV'][gen_idx][10]
        Kd = (2 / 40) * gen[7] / gen[2]
        Ppu = gen[4] / gen[2]

        # Calculate q0, T1, T2, and Dg
        # From linearization
        q0 = (Ppu + (A_t * q_nl)) / (A_t - (0 * Dt)) 
        T1 = (q0 - q_nl) * Tw
        T2 = q0 * Tw / 2
        Dg = q0 * Dt

        print(f"Generator {gen_idx}: T1={T1}, T2={T2}, A_t={A_t}, Tr={Tr}, r={r}, Tf={Tf}, Tg={Tg}, R={R}, H={H}, Kd={Kd}, Dg={Dg}, q0={q0}, q_nl={q_nl}")

        # Create Gp for this generator
        Gp, _ = create_Gp(T1, T2, A_t, Tr, r, Tf, Tg, R, H, Kd, Dg, 0)
        Gp_list.append(Gp)  # Add Gp to the list

    return Gp_list  # Return the list of Gp for all generators







    # # Plot the Bode magnitude plot
    # plt.figure()
    # #plt.semilogx(Wf, magF, label="Frequency dependence magnitude", color='black')  # Magnitude in absolute terms
    # plt.semilogx(w, mag, label="G_req-n magnitude", color='yellow')  # Magnitude in absolute terms
    # plt.semilogx(w, 0.9*mag, label="G_req-n magnitude with reduction factor", color='purple')  # Magnitude in absolute terms
    # plt.semilogx(w_k2a, mag_k2a, label="G_req-n magnitude K2A", color='blue')  # Magnitude in absolute terms
    # plt.semilogx(w_k2a, 0.9*mag_k2a, label="G_req-n magnitude K2A with reduction factor", color='green')  # Magnitude in absolute terms
    # plt.semilogx(w_req, mag_req, label="Requirement magnitude", linestyle='dashed', color='darkred')  # Magnitude in absolute terms
    # #plt.semilogx(f_load_list, Pxx_load_listDiv, label="Load FFT")
    # plt.title('Bode Magnitude Plot')
    # plt.xlabel('Frequency (rad/s)')
    # plt.ylabel('Magnitude (abs)')
    # #plt.xlim(0.01, 10)
    # plt.ylim(0, 15)
    # plt.legend()
    # plt.xlim(0.001, 10)
    # plt.grid()
    # plt.show()

def returnSystemTf(model_data):
    model = model_data.load()
    ########### Parameters for the system ################
    scalingfactor = 0.95

    ########### With TSO parameters ######################
    DeltaP_FCRN = 600  # MW
    Deltaf_FCRN = 0.1  # Hz
    f_0 = 50  # Hz
    S_n_FCRN = 42000  # MW

    ############ With K2A parameters #####################
    DeltaP_FCRN_K2a = 60  # MW
    S_n_FCRN_k2a = 0  # MW


    DeltaP_FCRN_K2a = 0  # MW
    S_n_FCRN_k2a = 0  # MW
    

    FCR_gens=[]



    for gen_idx, gen in enumerate(model['generators']['GEN'][1:], start=1):  # Skip the first entry (header)
        R = model['gov']['HYGOV'][gen_idx][2]
        H = gen[6]
        Dt = model['gov']['HYGOV'][gen_idx][10]
        Kd = (2 / 40) * gen[7] / gen[2]
        S_n_FCRN_k2a += gen[2]
        FCR_gens.append((Deltaf_FCRN/f_0)/(R/gen[2]))
    
    DeltaP_FCRN_K2a=sum(FCR_gens)


    print("DeltaP:", DeltaP_FCRN_K2a)
    print("S_n:", S_n_FCRN_k2a)

    Hsys = calculateSystemInertia(model)  # s
    GFCR_N = (DeltaP_FCRN / Deltaf_FCRN) * (f_0 / S_n_FCRN) * 1 / (2 * 4.524 * s + 0.01 * f_0)

    GFCR_N_k2a = (DeltaP_FCRN_K2a / Deltaf_FCRN) * (f_0 / S_n_FCRN_k2a) * 1 / (2 * Hsys * s)# + 0.05 * f_0)

    Stab_line=GFCR_N_k2a*2.31
    numeratorStab, denominatorStab = sp.fraction(Stab_line)
    numerator_coeffs_stab = [float(c) for c in sp.Poly(numeratorStab, s).all_coeffs()]
    denominator_coeffs_stab = [float(c) for c in sp.Poly(denominatorStab, s).all_coeffs()]
    tf_stab = TransferFunction(numerator_coeffs_stab, denominator_coeffs_stab)

    omega = np.logspace(-5, 3, 500)  # Frequency range for Bode plot
    # Compute the Bode plot
    # w_stab, mag_stab, phase_stab = bode(tf_stab, w=omega)
    # mag_stab = (10 ** (mag_stab / 20)) 
    # print("Stab line:", mag_stab)

    # df_stab = pd.DataFrame({'w_stab': w_stab, 'mag_stab': mag_stab})
    # outdir = r'C:\Users\haral\PycharmProjects\ProsjektOppgaveTOPS\Figures\FrequencyPlots'
    # os.makedirs(outdir, exist_ok=True)
    # df_stab.to_csv(os.path.join(outdir, 'stabilityLineK2A.csv'), index=False)

    
    # Get the list of transfer functions (F)
    F_list = checkGetTf(model)

    # Initialize lists to store results for each F
    G_req_n_list = []
    G_req_n_k2a_list = []
    G_req_n_reduction_list = []

    for F in F_list:
        # Simplify F
        F = sp.simplify(F)
        F_static_gain = sp.simplify(F.subs(s, 0))
        print("F_static_gain:", F_static_gain)


        # Calculate G_req_n, G_req_n_k2a, and G_req_n_reduction for this F
        G_req_n = scalingfactor * GFCR_N / (1 + GFCR_N * F / F_static_gain)
        G_req_n_reduction = 0.9 * G_req_n
        G_req_n_k2a = scalingfactor * GFCR_N_k2a / (1 + GFCR_N_k2a * F / F_static_gain)

        # Simplify the results
        G_req_n = sp.simplify(G_req_n)
        G_req_n_reduction = sp.simplify(G_req_n_reduction)
        G_req_n_k2a = sp.simplify(G_req_n_k2a)

        # Append the results to the respective lists
        G_req_n_list.append(G_req_n)
        G_req_n_k2a_list.append(G_req_n_k2a)
        G_req_n_reduction_list.append(G_req_n_reduction)
    return G_req_n_list, G_req_n_k2a_list, G_req_n_reduction_list, FCR_gens



def calculateSystemTf(model_data):
    # Get the lists of transfer functions from returnSystemTf
    G_req_n_list, G_req_n_k2a_list, G_req_n_reduction_lis, FCR_gens= returnSystemTf(model_data)

    # Initialize the frequency range for Bode plots
    omega = np.logspace(-5, 3, 500)

    # Initialize lists to store results
    w_list = []
    mag_list = []
    w_k2a_list = []
    mag_k2a_list = []
    w_req_list = []
    mag_req_list = []

    d=TransferFunction([1], [70, 1])
    w_req, actualReq, phaseReq=bode(d, w=omega)
    actualReq = (10 ** (actualReq / 20))  # Convert from dB to absolute magnitude

    # Create a new figure for the plot

    #Process each transfer function in G_req_n_list
    for idx, G_req_n in enumerate(G_req_n_list, start=1):
        numerator, denominator = sp.fraction(G_req_n)
        numerator_coeffs = [float(c) for c in sp.Poly(numerator, s).all_coeffs()]
        denominator_coeffs = [float(c) for c in sp.Poly(denominator, s).all_coeffs()]
        tf_n = TransferFunction(numerator_coeffs, denominator_coeffs)

        # Compute the Bode plot
        w, mag, phase = bode(tf_n, w=omega)
        mag = (10 ** (mag / 20))  # Convert from dB to absolute magnitude

        # Append results to lists
        w_list.append(w)
        mag_list.append(mag)

    #     # Plot the magnitude
    #     plt.semilogx(w, mag, label=f"G_req_n {idx} magnitude", alpha=0.7)

    # Process each transfer function in G_req_n_k2a_list
    for idx, G_req_n_k2a in enumerate(G_req_n_k2a_list, start=1):
        numerator_k2a, denominator_k2a = sp.fraction(G_req_n_k2a)
        numerator_coeffs_k2a = [float(c) for c in sp.Poly(numerator_k2a, s).all_coeffs()]
        denominator_coeffs_k2a = [float(c) for c in sp.Poly(denominator_k2a, s).all_coeffs()]
        tf_n_k2a = TransferFunction(numerator_coeffs_k2a, denominator_coeffs_k2a)

        # Compute the Bode plot
        w_k2a, mag_k2a, phase_k2a = bode(tf_n_k2a, w=omega)
        mag_k2a = (10 ** (mag_k2a / 20))  # Convert from dB to absolute magnitude

        # Append results to lists
        w_k2a_list.append(w_k2a)
        mag_k2a_list.append(mag_k2a)

        # Plot the magnitude
        plt.semilogx(w_k2a, mag_k2a, label=f"G_req_n_k2a {idx} magnitude", alpha=0.7)


    # Return all results as lists
    return w_list, mag_list, w_k2a_list, mag_k2a_list, w_req_list, mag_req_list, w_req, actualReq, FCR_gens

#calculateSystemTf(model_data)

#For making magnitude bode plot (rad/s on the x-axis), in absolute magnitude:
def makeFFT(timeseries, timestep):
    N = len(timeseries)  # Number of samples
    T = timestep  # Sampling time (timestep from simulation)
    fs = 1 / T  # Sampling frequency
    print(fs)
    fstep=fs/N   #Frequency interval
    t=np.linspace(0, (N-1)*timestep, N)

    # Compute FFT
    fft_values = np.fft.fft(timeseries)
    f=np.fft.fftfreq(N, d=timestep)
    fft_mag=np.abs(fft_values)/N  #Magnitude of FFT

    f_plot= f[0:int(N/2+1)]
    fft_mag_plot=2*fft_mag[0:int(N/2+1)]  #Only plot the first half of the spectrum

    # Use only the strictly positive part
    f_plot = f_plot[:-1]
    fft_mag_plot = fft_mag_plot[:-1]

    f_plot_rad=f_plot*2*np.pi  #Convert to rad/s

    # # Fix Nyquist component (only if N is even)
    # if N % 2 == 0:
    #     fft_mag_plot[-1] /= 2  # Undo doubling for Nyquist frequency

    print("Is frequency sorted?", np.all(np.diff(f_plot) > 0))
    # # plot
    # fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
    # ax1.plot(t, timeseries)
    # ax1.set_xlabel('Time (s)')
    # ax1.set_ylabel('Amplitude')
    #
    # ax2.plot(f_plot, fft_mag_plot)
    # ax2.set_xlabel('Frequency (Hz)')
    # ax2.set_ylabel('Magnitude')
    # plt.legend()
    # plt.show()
    return f_plot_rad, fft_mag_plot

def makeWelchFFT(timeseries, timestep, nperseg=40000, scaling='spectrum'):
    f_plot, fft_magplot=makeFFT(timeseries, timestep)
    N=len(timeseries)
    T=timestep
    fs=1/T
    f, Pxx = welch(timeseries, fs, nperseg=nperseg, scaling=scaling, noverlap=nperseg//1.33)
    Pxx = np.sqrt(Pxx)* np.sqrt(2)            #going from rms V**2 to just V (magnitude)
    f=f*2*np.pi  #Convert to rad/s
    f_plot=f_plot*2*np.pi

    #delta_f=f[1]-f[0]         #frequency resolution
    print("Datapoints per segment Welch:",nperseg)
    # plt.semilogx(f, Pxx, label="Welch")
    # plt.semilogx(f_plot, fft_magplot, label="FFT")
    # plt.xlabel('frequency [rad/s]')
    # plt.ylabel('Magnitude')
    # plt.legend()
    # plt.show()
    return f, Pxx



def TransferFunctionFromIPOP(inputsignal, outputsignal, timestep1, timestep2):
    # Calculate the FFT of the input signal
    frequencies, input_fft = makeFFT(inputsignal, timestep1)
    # Calculate the FFT of the output signal
    _, output_fft = makeFFT(outputsignal, timestep2)
    # Calculate the transfer function
    print("input_FFT:", input_fft)
    print("lengthInput", len(input_fft))
    print("output_FFT:", output_fft)
    print("lengthOutput", len(output_fft))

    transfer_function = output_fft / input_fft
    print("TransferFunction:", transfer_function)
    return frequencies, transfer_function

def calculateDisturbanceMagnitude(model_data, inputsignal, outputsignal, timestep1, timestep2):
    frequencies, Measuredtransfer_function = TransferFunctionFromIPOP(inputsignal, outputsignal, timestep1, timestep2)
    print(Measuredtransfer_function)
    _,systemTf=returnSystemTf(model_data)  #Get the K2a system transfer function
    print(systemTf)
    numerator, denominator = sp.fraction(systemTf)

    numerator_coeffs = [float(c) for c in sp.Poly(numerator, s).all_coeffs()]  # Had some issues with symbolic representations
    denominator_coeffs = [float(c) for c in sp.Poly(denominator, s).all_coeffs()]
    tf_n = TransferFunction(numerator_coeffs, denominator_coeffs)
    w, mag, phase = bode(tf_n, w=frequencies)
    mag = (10 ** (mag / 10))
    mag_Measured=np.abs(Measuredtransfer_function)
    # Calculate the magnitude of the disturbance
    D=mag_Measured/mag
    return frequencies, D, mag

# def compareFreqandDisturbance(disturbanceMagnitude, systemTfMagnitude, disturbancefrequencies):
#     plt.figure()
#     plt.semilogx(disturbancefrequencies, disturbanceMagnitude, label="Disturbance")
#     plt.semilogx(disturbancefrequencies, systemTfMagnitude, label="System Transfer Function")
#     plt.title('Bode Magnitude Plot')
#     plt.xlabel('Frequency (rad/s)')
#     plt.ylabel('Magnitude (abs)')
#     plt.legend()
#     plt.grid()
#     plt.show()






def plot_frequency_pdf(folder_path, frequency_column_name):
    """
    Reads all CSV files in a folder, extracts frequency values (divided by 4),
    and plots a probability density function (PDF) using Pandas KDE.

    Parameters:
    - folder_path: Path to the folder containing CSV files.
    - frequency_column_name: The name of the column containing frequency values.

    Returns:
    - freq_values: The combined and corrected frequency data from all CSV files.
    """
    all_frequencies = []  # List to store all corrected frequency values

    # Loop through all CSV files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):  # Process only CSV files
            file_path = os.path.join(folder_path, filename)

            try:
                # Read the CSV file
                df = pd.read_csv(file_path)

                # Check if the specified column exists
                if frequency_column_name in df.columns:
                    # Divide by 4 and store the corrected frequencies
                    corrected_frequencies = df[frequency_column_name].dropna().values / 4
                    all_frequencies.extend(corrected_frequencies)
                else:
                    print(f"Warning: Column '{frequency_column_name}' not found in {filename}")

            except Exception as e:
                print(f"Error reading {filename}: {e}")

    # Convert to Pandas Series
    freq_series = pd.Series(all_frequencies)

    # Check if we have data to plot
    if freq_series.empty:
        print("No frequency data found.")
        return None
    print(len(freq_series))

    # # Plot PDF using Pandas KDE
    # plt.figure(figsize=(10, 5))
    # ax = plt.gca()  # Get current axis
    # freq_series.plot(kind='kde', ax=ax, label="Corrected Frequency PDF", color='blue')
    #
    # # Formatting the plot
    # plt.xlabel("Corrected Frequency (Hz)")
    # plt.ylabel("Probability Density")
    # plt.title("Probability Density Function of Corrected Frequencies")
    # plt.legend()
    # plt.grid(True)

    # Show the plot
    #plt.show()

    return freq_series.values  # Return the raw corrected frequency data if needed

def fitNormalDistribution(freq_series):
    mean=np.mean(freq_series)
    std_dev=np.std(freq_series)
    #Plotting the normal distribution
    x = np.linspace(mean - 3 * std_dev, mean + 3 * std_dev, 1000)
    y = (1 / (std_dev * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std_dev) ** 2)
    plt.plot(x, y, label='Normal Distribution of Frequency')
    plt.legend()
    plt.show()
    return mean, std_dev

# Example usage:
# plot_frequency_pdf("path/to/your/folder", "FrequencyColumnName")


# Example usage:
#plot_frequency_pdf("C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData", "Frequency [Hz*4]")




#PlotFreqData('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous (1)/Outage-anonymous.xlsx', 'FI south:Frequency')

def interpolate_exponential(function, x_values, y_values):
    return curve_fit(function, x_values, y_values)

def plotPSD(data, dataname):
    f1, Pxx1 = periodogram(data, 50)
    Pxx1_dB = 20 * np.log10(abs(Pxx1))
    # plt.plot(f1, Pxx1_dB, label=dataname)
    # plt.xlabel('Frequency [Hz]')
    # plt.ylabel('PSD [dB/Hz]')
    # plt.xlim(0, 2)
    # plt.legend()
    # plt.show()
    return f1, Pxx1

# def PlotFrequencyVariance(loadValues, freqValues, model_data):
    G_req_n, G_req_n_k2a, G_req_n_reduction=returnSystemTf(model_data)
    numerator_k2a, denominator_k2a = sp.fraction(G_req_n_k2a)
    numerator_coeffs_k2a = [float(c) for c in sp.Poly(numerator_k2a, s).all_coeffs()]  # Had some issues with symbolic representations
    denominator_coeffs_k2a = [float(c) for c in sp.Poly(denominator_k2a, s).all_coeffs()]
    tf_n_k2a = TransferFunction(numerator_coeffs_k2a, denominator_coeffs_k2a)
    mean, std_dev=fitNormalDistribution(freqValues)
    print("Simulated frequency mean: ", mean, "\nStandard deviation: ", std_dev)
    plotPSD(freqValues, "Simulated Frequency PSD")
    f1, Pxx1=plotPSD(loadValues, "Simulated load PSD")
    print(int(50//len(loadValues)))
    print(f1)
    omega=np.linspace(0, 2*np.pi*50/2, int((len(loadValues)+1)/2))
    w_k2a, mag_k2a, phase_k2a = bode(tf_n_k2a, w=omega)
    w_k2a=w_k2a/(2*np.pi)     #Convert from rad/s to Hz
    mag_k2a = (10 ** (mag_k2a / 20))  # Convert from dB to absolute magnitude
    print(w_k2a)
    print(f1)
    frequencyVariance=mag_k2a*np.sqrt(Pxx1)
    print("Frequency Variance: ", frequencyVariance)
    plt.plot(f1, frequencyVariance, label="Frequency Variance")
    plt.plot(w_k2a, mag_k2a)
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('PSD [Hz/Hz]')
    plt.xlim(0, 2)
    plt.legend()
    plt.show()


def plotACFs(freqdata):
    data1 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_1.csv')
    data2 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_2_pelton.csv')
    data3 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_3.csv')
    data4 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_4.csv')
    data5 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_5.csv')
    data6 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_6.csv')
    data7 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_7.csv')
    data8 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_8.csv')
    data9=freqdata

    #Plotting the acf of all the datasets in the same plot for 200 seconds (10000 lags)
    acf_val_freq1=acf(data1['Frequency [Hz*4]'][0:len(freqdata)], nlags=10000)
    acf_val_freq2=acf(data2['Frequency [Hz*4]'][0:len(freqdata)], nlags=10000)
    acf_val_freq3=acf(data3['Frequency [Hz*4]'][0:len(freqdata)], nlags=10000)
    acf_val_freq4=acf(data4['Frequency [Hz*4]'][0:len(freqdata)], nlags=10000)
    acf_val_freq5=acf(data5['Frequency [Hz*4]'][0:len(freqdata)], nlags=10000)
    acf_val_freq6=acf(data6['Frequency [Hz*4]'][0:len(freqdata)], nlags=10000)
    acf_val_freq7=acf(data7['Frequency [Hz*4]'][0:len(freqdata)], nlags=10000)
    acf_val_freq8=acf(data8['Frequency [Hz*4]'][0:len(freqdata)], nlags=10000)
    acf_val_freq9=acf(data9, nlags=10000)

    timevalues=np.arange(len(acf_val_freq9))*0.02  #Seconds

    plt.plot(timevalues, acf_val_freq1, color='blue', alpha=0.2)
    plt.plot(timevalues, acf_val_freq2, color='blue', alpha=0.2)
    plt.plot(timevalues, acf_val_freq3, color='blue', alpha=0.2)
    plt.plot(timevalues, acf_val_freq4, color='blue', alpha=0.2)
    plt.plot(timevalues, acf_val_freq5, color='blue', alpha=0.2)
    plt.plot(timevalues, acf_val_freq6, color='blue', alpha=0.2)
    plt.plot(timevalues, acf_val_freq7, color='blue', alpha=0.2)
    plt.plot(timevalues, acf_val_freq8, color='blue', label='Real Frequency Data Autocorrelation', alpha=0.2)
    plt.plot(timevalues, acf_val_freq9, color='red', alpha=1.0, label='Simulated Frequency Data Autocorrelation')
    
    plt.legend()
    plt.title('Autocorrelation of all datasets and Exponential Functions')
    plt.xlabel('Time [s]')
    plt.ylabel('Autocorrelation')
    plt.show()


def ExpectedFrequency(LoadChange, SysGoverningGain, initialLoad):
    deltaf_pu=(-1/SysGoverningGain)*(LoadChange/initialLoad)
    return 50*(1+deltaf_pu)  #Convert to Hz








