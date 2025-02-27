import pandas as pd
import matplotlib.pyplot as plt
import tops.dynamic as dps
from examples.interfaces.results_events import ResultKeeper, Events
from tops.simulator import Simulator
import examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
#import tops.ps_models.k2a as model_data
import numpy as np
from collections import defaultdict
import time
import sys
from sympy.abc import s
import sympy as sp
from scipy.signal import TransferFunction, bode


def runDynSimulation(model, t_end, timestep, H, Tf, Tr, Tg, Tw, R, r):
    #model_data = __import__(model)
    model = model_data.load()
    #model['loads'] = {'DynamicLoad': model['loads']}
    model['loads'] = {'ConstantPowerLoad': model['loads']}

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
    '''if 'pss' in model and 'STAB1' in model['pss']:
        for stab in model['pss']['STAB1'][1:]:
            stab[2] = K
            stab[3] = T
            stab[4] = T_1
            stab[5] = T_2
            stab[6] = T_3
            stab[7] = T_4
            stab[8] = H_lim'''
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    # ps.ode_fun(0, ps.x0)
    sim = Simulator(ps, dt=timestep, t_end=t_end)
    res_keeper = ResultKeeper(sim)
    events = Events(sim, [
        (17.2, ('dynload', 'g_setp', 1.09, 0)),
        (17.2, ('dynload', 'b_setp', 0.26, 0)),
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

    time, freq=GetFreqData('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous (1)/Outage-anonymous.xlsx', 'FI south:Frequency')
    freq=freq[0:2000]
    time=time[0:2000]
    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/Taajuusdata2024-05-05.csv')
    freq1=data['Value'][156800:158100]
    time1=[]
    for i in range(len(freq1)):
        time1.append(i/10)

    # Calculate the average speed
    df['average_speed'] = df[speed_columns].mean(axis=1)
    df['avg_freq'] = (df['average_speed'] * 50 / (2 * np.pi)) + 50.04
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

#For k2a
runDynSimulation(model_data, 60, 5e-3, 3 , 0.1, 7, 0.6, 3.7, 0.14, 1.8) #, 20, 10, 0.3, 0.9, 0.05, 0.07, 0.04)

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
    Gs = (sp.exp(-s * Tdel) / (1 + s * Tg))  #SGate Servomotor
    Gj = 1 / (2 * H * s + Kd)
    Gp = Gt * Gc * Gs + Dg
    S = 1 / (1 + Gj * Gp)
    G0 = -Gj * S
    return Gp, Gj

def checkGetTf(model):
    ps=dps.PowerSystemModel(model=model)
    #s = sp.symbols('s')
    Tw = model['gov']['HYGOV'][1][8]
    q_nl=model['gov']['HYGOV'][1][9]
    A_t=model['gov']['HYGOV'][1][7]
    Tr=model['gov']['HYGOV'][1][5]
    r=model['gov']['HYGOV'][1][3]
    Tf=model['gov']['HYGOV'][1][4]
    Tg=model['gov']['HYGOV'][1][6]
    R=model['gov']['HYGOV'][1][2]
    H=model['generators']['GEN'][1][6]
    Dt=model['gov']['HYGOV'][1][10]
    Kd=(2/40)*model['generators']['GEN'][1][7]/model['generators']['GEN'][1][2]
    w0=2*np.pi*50
    Ppu=model['generators']['GEN'][1][4]/model['generators']['GEN'][1][2]
    q0 = (Ppu + (A_t * q_nl)) / (A_t - (w0 * Dt))    #From linearization
    T1=(q0-q_nl)*Tw
    T2=q0*Tw/2
    Dg=q0*Dt
    print('T1=',T1, 'T2=',T2, 'At=',A_t, 'Tr=',Tr, 'r=',r, 'Tf=',Tf, 'Tg=',Tg, 'R=',R, 'H=',H, 'Kd=',Kd, 'Dg=',Dg, 'q0=',q0, 'qnl=',q_nl)
    Gp, Gj=create_Gp(T1, T2, A_t, Tr, r, Tf, Tg, R, H, Kd, Dg, 0)
    return Gp, Gj


checkGetTf(model_data.load())
def  calculateSystemTf():
    model=model_data.load()
    #scaledNSAtoK2A=3600/42000
    scalingfactor=0.95
    #With TSO parameters
    DeltaP_FCRN=600                                #MW
    DeltaP_FCRD=1450                               #MW
    Deltaf_FCRN=0.1                                #Hz
    Deltaf_FCRD=0.4                                #Hz
    f_0=50                                         #Hz
    S_n_FCRN=42000                                 #MW
    S_n_FCRD=42000                                 #MW

    #With K2A parameters
    DeltaP_FCRN_K2a=600                                #MW    Not sure how to scale the amount delivered for FCR-N
    DeltaP_FCRD_K2a=900                               #MW   Rated power of scaling unit
    S_n_FCRN_k2a=3600
    S_n_FCRD_k2a=3600
    Hsys=calculateSystemInertia(model)
    GFCR_N=(DeltaP_FCRN/Deltaf_FCRN)*(f_0/S_n_FCRN)*1/(2*4.524*s+0.01*f_0)

    GFCR_D=(DeltaP_FCRD/Deltaf_FCRD)*(f_0/S_n_FCRD)*1/(2*4.524*s+0.01*f_0)
    GFCR_N_k2a=(DeltaP_FCRN_K2a/Deltaf_FCRN)*(f_0/S_n_FCRN_k2a)*1/(2*Hsys*s+0.01*f_0)
    GFCR_D_k2a=(DeltaP_FCRD_K2a/Deltaf_FCRD)*(f_0/S_n_FCRD_k2a)*1/(2*Hsys*s+0.01*f_0)
    F, Gj= checkGetTf(model)

    F=sp.simplify(F)
    F_numerator, F_denominator = sp.fraction(F)
    tfFF=TransferFunction([float(c) for c in sp.Poly(F_numerator, s).all_coeffs()], [float(c) for c in sp.Poly(F_denominator, s).all_coeffs()])

    G_req_n=scalingfactor*GFCR_N/(1+GFCR_N*F/7.14)
    G_req_n_reduction=0.9*scalingfactor*GFCR_N/(1+GFCR_N*F/7.14)
    G_req_d=scalingfactor*GFCR_D/(1+GFCR_D*F/7.14)

    G_req_n_k2a=scalingfactor*GFCR_N_k2a/(1+GFCR_N_k2a*F/7.14)
    G_req_d_k2a=scalingfactor*GFCR_D_k2a/(1+GFCR_D_k2a*F/7.14)

    #print(sp.pretty(G_req_n))
    G_req_n_simple = sp.simplify(G_req_n)
    G_req_n_simpleReduction = sp.simplify(G_req_n_reduction)
    G_req_d_simple = sp.simplify(G_req_d)

    G_req_n_k2a_simple = sp.simplify(G_req_n_k2a)
    G_req_d_k2a_simple = sp.simplify(G_req_d_k2a)

    #print(sp.pretty(G_req_n_simple))
    #print(G_req_n_simple)
    #print(sp.factor(G_req_n_simple))
    #print(G_req_d_simple)

    print(G_req_n_k2a_simple)
    #print(G_req_d_k2a_simple)
    numerator, denominator = sp.fraction(G_req_n_simple)
    numerator_simple, denominator_simple = sp.fraction(G_req_n_simpleReduction)
    numerator_d,denominator_d=sp.fraction(G_req_d_simple)

    numerator_k2a, denominator_k2a = sp.fraction(G_req_n_k2a_simple)
    numerator_d_k2a,denominator_d_k2a=sp.fraction(G_req_d_k2a_simple)

    numerator_coeffs = [float(c) for c in sp.Poly(numerator, s).all_coeffs()]    #Had some issues with symbolic representations
    denominator_coeffs = [float(c) for c  in sp.Poly(denominator, s).all_coeffs()]
    numerator_coeffs_d = [float(c) for c in sp.Poly(numerator_d, s).all_coeffs()]    #Had some issues with symbolic representations
    denominator_coeffs_d = [float(c) for c  in sp.Poly(denominator_d, s).all_coeffs()]

    numerator_coeffs_k2a = [float(c) for c in sp.Poly(numerator_k2a, s).all_coeffs()]    #Had some issues with symbolic representations
    denominator_coeffs_k2a = [float(c) for c  in sp.Poly(denominator_k2a, s).all_coeffs()]
    numerator_coeffs_d_k2a = [float(c) for c in sp.Poly(numerator_d_k2a, s).all_coeffs()]    #Had some issues with symbolic representations
    denominator_coeffs_d_k2a = [float(c) for c  in sp.Poly(denominator_d_k2a, s).all_coeffs()]


    tf_n = TransferFunction(numerator_coeffs, denominator_coeffs)
    tf_n_reduction=TransferFunction([float(c) for c in sp.Poly(numerator_simple, s).all_coeffs()], [float(c) for c in sp.Poly(denominator_simple, s).all_coeffs()])


    tf_d = TransferFunction(numerator_coeffs_d, denominator_coeffs_d)

    tf_n_k2a = TransferFunction(numerator_coeffs_k2a, denominator_coeffs_k2a)
    tf_d_k2a = TransferFunction(numerator_coeffs_d_k2a, denominator_coeffs_d_k2a)

    omega=np.logspace(-5,3,500)

    w, mag, phase = bode(tf_n, w=omega)
    mag = (10 ** (mag / 20))  # Convert from dB to absolute magnitude

    w_reduction, mag_reduction, phase_reduction = bode(tf_n_reduction, w=omega)
    mag_reduction = (10 ** (mag_reduction / 20))  # Convert from dB to absolute magnitude

    w2d, mag2d, phase2=bode(tf_d, w=omega)
    mag2 = (10 ** (mag2d / 20))  # Convert from dB to absolute magnitude

    w_k2a, mag_k2a, phase_k2a = bode(tf_n_k2a, w=omega)
    mag_k2a = (10 ** (mag_k2a / 20))  # Convert from dB to absolute magnitude

    wd_k2a, magd_k2a, phased_k2a = bode(tf_d_k2a, w=omega)
    magd_k2a = (10 ** (magd_k2a / 20))  # Convert from dB to absolute magnitude

    Wf,magF, phaseF=bode(tfFF, w=omega)
    magF = (10 ** (magF / 20))  # Convert from dB to absolute magnitude

    tf2_req = (TransferFunction([70,1], [1]))
    w_req, mag_req, phase_req = bode(tf2_req, w=omega)
    #mag_req = 1 / mag_req
    mag_req = 10 ** (mag_req / 20)  # Convert from dB to absolute magnitude

    load_periodogram_df = pd.read_csv("C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/examples/sde_solver/loadperiodogram.csv")
    # Extract the data into two lists
    f_load_list = load_periodogram_df.iloc[:, 0].tolist()
    #print(f_load_list)
    Pxx_load_list = load_periodogram_df.iloc[:, 1].tolist()
    #print(Pxx_load_list)
    Pxx_load_listDiv = [1/value for value in Pxx_load_list]





    # Plot the Bode magnitude plot
    plt.figure()
    #plt.semilogx(Wf, magF, label="Frequency dependence magnitude", color='black')  # Magnitude in absolute terms
    #plt.semilogx(w, mag, label="G_req-n magnitude", color='blue')  # Magnitude in absolute terms
    #plt.semilogx(w, 0.9*mag, label="G_req-n magnitude with reduction factor", color='green')  # Magnitude in absolute terms
    plt.semilogx(w_k2a, mag_k2a, label="G_req-n magnitude K2A", color='blue')  # Magnitude in absolute terms
    plt.semilogx(w_k2a, 0.9*mag_k2a, label="G_req-n magnitude K2A with reduction factor", color='green')  # Magnitude in absolute terms
    #plt.semilogx(wd_k2a, magd_k2a, label="G_req-d magnitude K2A with reduction factor", color='green')  # Magnitude in absolute terms
    plt.semilogx(w_req, mag_req, label="Requirement magnitude", linestyle='dashed', color='darkred')  # Magnitude in absolute terms
    #plt.semilogx(f_load_list, Pxx_load_listDiv, label="Load FFT")
    plt.title('Bode Magnitude Plot')
    plt.xlabel('Frequency (rad/s)')
    plt.ylabel('Magnitude (abs)')
    #plt.xlim(0.01, 10)
    plt.ylim(0, 15)
    plt.legend()
    plt.xlim(0.001, 10)
    plt.grid()
    plt.show()





calculateSystemTf()




# Assuming measured_time and measured_freq are your measured data
#best_params, best_diff = compare_simulation_pss_only(30, 0.5, param_bounds, measured_time, measured_freq)
#print("Best Parameters:", best_params)
#print("Best Difference:", best_diff)



#PlotFreqData('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous (1)/Outage-anonymous.xlsx', 'FI south:Frequency')

