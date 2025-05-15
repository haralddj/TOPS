import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np
import wesanderson as ws
import tops.examples.user_models.user_lib.MyTools.Functions as fc
import os  # Add this import to handle file paths



colors=ws.film_palette('Darjeeling limited')

if __name__ == '__main__':

    # Load model
    import tops.examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    #import src.tops.ps_models.k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()
    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))


    t_end = 60
    x_0 = ps.x_0.copy()
    print(ps.v0)

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, ps.v_0, t_end, max_step=0.02)
    # sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, ps.v0, t_end, max_step=time_step)


    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    sc_bus_idx = ps.gen['GEN'].bus_idx_red['terminal'][0]
    line_mdl = ps.lines['Line']


    def p_bus_7(x, v):
        return -ps.s_n * (line_mdl.p_to(x, v)[1] + line_mdl.p_from(x, v)[2] + line_mdl.p_from(x, v)[3])

    def q_bus_7(x, v):
        return -ps.s_n * (line_mdl.q_to(x, v)[1] + line_mdl.q_from(x, v)[2] + line_mdl.q_from(x, v)[3])
    
    def p_bus_9(x,v):
        return -ps.s_n * (line_mdl.p_to(x, v)[4] + line_mdl.p_to(x, v)[5] + line_mdl.p_from(x, v)[6])


    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        if 17.2 <= t: 
            ps.loads['DynamicLoad'].set_input('g_setp', 1.239, 0)

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        
        # Store result
        res['t'].append(t)
        res['freq'].append(ps.gen['GEN'].speed(x, v).copy()*50+50.04)
        #res['freq'].append(((ps.gen['GEN'].speed(x, v).copy() * 50)) + 50.04)

        res['v'].append(v.copy())
        res['p_bus_7'].append(p_bus_7(x, v).copy())
        res['q_bus_7'].append(q_bus_7(x, v).copy())
        res['p_bus_9'].append(p_bus_9(x, v).copy())

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

    ##################################################
    #############       Plotting       ###############
    ##################################################
    # Ensure the output directory exists
    output_dir = 'C:/Users/haral/.vscode/TOPS_HeltNy/TOPS/ResultaterFigurer/DynamiskTuning'
    os.makedirs(output_dir, exist_ok=True)

    #Frequency plot

    timeOutage, freqOutage=fc.GetFreqData('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous_(1)/Outage-anonymous.xlsx', 'FI south:Frequency')
    freqOutage=freqOutage[0:2000]
    timeOutage=timeOutage[0:2000]
    avg_freq = [np.mean(freq) for freq in res['freq']]


    fig = plt.figure()
    plt.plot(res['t'], avg_freq, label="Simulated Frequency", color=colors[2])

    plt.plot(timeOutage[0:1500], freqOutage[:1500], label="Real Frequency", color=colors[0])
    plt.xlabel('Time [s]')
    plt.ylabel('Frequency [Hz]')
    plt.title('Real vs. Simulated Frequency')
    plt.legend()
    #plt.savefig(os.path.join(output_dir, 'frequency_plot_finishedTuning.pdf'))  # Save as PDF
    #Load Power plot

    plt.figure()
    #plt.plot(res['t'], abs(np.asarray(res['v'])), label="v", color=colors[1])
    #plt.plot(res['t'], np.abs(res['p_bus_9']), label="Real Power at bus 9", color=colors[1])
    plt.plot(res['t'], np.abs(res['p_bus_7']), label="Real Power at bus 7", color=colors[3])
    #plt.plot(res['t'], np.abs(res['q_bus_7']), label="Reactive Power at bus 7")
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Power [MW]')
    plt.title('Real Power of Load at bus 7')
    #plt.savefig(os.path.join(output_dir, 'load_power_plotTuned.pdf'))  # Save as PDF
    
    plt.show()
    