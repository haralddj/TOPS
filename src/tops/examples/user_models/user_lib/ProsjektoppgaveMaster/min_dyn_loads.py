import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np
import pandas as pd
import tops.examples.user_models.user_lib.MyTools.Functions as fc



if __name__ == '__main__':

    # Load model
    import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    #import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_sm_ib as model_data
    #import src.tops.ps_models.n44 as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))
    print(ps.sys_data)

    t_end = 100
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, ps.v_0, x_0, t_end, max_step=0.02)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]
    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))
        # Short circuit
        if 17.2 <= t: #g_setp refers to the real part of the admittance of the load, b_setp refers to the imaginary part of the admittance
            ps.loads['DynamicLoad'].set_input('g_setp', 1.45, 0)
            #ps.loads['DynamicLoad'].set_input('b_setp', 0, 0)


        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['time'].append(t)
        #res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['freq'].append(ps.gen['GEN'].speed(x, v).copy()*50+50.03)
        res['v'].append(v.copy())
        #res['gen_I'].append(ps.gen['GEN'].I(x, v).copy())
        #res['load_I'].append(ps.loads['DynamicLoad'].I(x, v).copy())
        res['load_P'].append(ps.loads['DynamicLoad'].P(x, v).copy())
        res['load_Q'].append(ps.loads['DynamicLoad'].Q(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    # Step 1: Read the CSV/excel file
    # data = pd.read_csv('/2024-05/Taajuusdata2024-05-01.csv')
    # freq_normal=data['Value'][17000:20000]
    # time_normal=[]
    # for i in range(len(freq_normal)):
    #     time_normal.append(i/10)
    # print(time_normal)
    # print(freq_normal)
    timevalues, frequency_data = fc.GetFreqData('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous_(1)/Outage-anonymous.xlsx', 'NO4a:Frequency')

    avg_freq=[np.mean(freq) for freq in res['freq']]
    avg_speed=[np.mean(speed) for speed in res['gen_speed']]
    # Create the plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

    # Plot frequency data
    ax1.plot(timevalues, frequency_data, label='Measured frequency')
    #ax1.plot(time_normal, freq_normal, label='Measured frequency')
    ax1.plot(res['time'], res['freq'], label='Simulated frequency')
    ax1.set_xlabel('Times [s]')
    ax1.set_ylabel('Frequency [Hz]')
    ax1.set_title('Measured vs Simulated Frequency')
    # ax1.legend()
    # ax1.grid(True)

    # # Plot load_p data
    # ax2.plot(res['time'], res['load_P'], label='Load P')
    # ax2.plot(res['time'], res['load_Q'], label='Load Q')
    # ax2.set_xlabel('Times [s]')
    # ax2.set_ylabel('Load P and Q [MW/MVAr)]')
    # ax2.set_title('Load P/Q over Time')
    # ax2.legend()
    # ax2.grid(True)

    plt.tight_layout()
    plt.show()














    '''fig = plt.figure()
    plt.plot(res['t'], np.abs(res['v']))
    plt.xlabel('Time [s]')
    plt.ylabel('Bus voltage')'''

    #fig = plt.figure()
    # Note: Generator current is higher than load current due to transformers
    '''plt.plot(res['t'], np.abs(res['gen_I']))
    plt.xlabel('Time [s]')
    plt.ylabel('Generator current [A]')

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['load_I']))
    plt.xlabel('Time [s]')
    plt.ylabel('Load current [A]')'''
    
    '''fig = plt.figure()
    plt.plot(res['t'], np.abs(res['load_P']))
    plt.xlabel('Time [s]')
    plt.ylabel('MW')

    fig = plt.figure()
    plt.plot(res['t'], np.abs(res['load_Q']))
    plt.xlabel('Time [s]')
    plt.ylabel('MVA')'''
    
    #plt.show()
    