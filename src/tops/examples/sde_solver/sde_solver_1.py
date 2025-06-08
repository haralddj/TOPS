import sys
from collections import defaultdict
import time
import tops.dynamic as dps
from tops.solvers_sde import EulerDAE_SDE
from scipy import signal
from tops.examples.user_models.user_lib.MyTools.Statistics import *

if __name__ == '__main__':

    # Load model
    import tops.examples.user_models.user_lib.k2aTunedToGrid.min_k2a as model_data
    model = model_data.load()
    model['loads'] = {'DynamicLoadFiltered':  [#model['loads']}
        model['loads'][0] + ['T_g', 'T_b'],
        *[row + [5, 5] for row in model['loads'][1:]]
    ]}

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    load_state_idx_g = ps.loads['DynamicLoadFiltered'].lpf_g.state_idx_global['x']
    load_state_idx_b = ps.loads['DynamicLoadFiltered'].lpf_b.state_idx_global['x']

    t_end = 600
    timeStep = 0.02
    # Solver
    sol = EulerDAE_SDE(ps.state_derivatives, ps.solve_algebraic, 0, ps.x_0, t_end, max_ste=0.02, dim_w = 2)

    def b_func(t, x, v):
        mat = np.zeros((len(sol.x), sol.dim_w))
        mat[load_state_idx_g[0], 0] = 0.005
        mat[load_state_idx_b[0], 0] = 0.003
        mat[load_state_idx_g[1], 1] = 0.005
        mat[load_state_idx_b[1], 1] = 0.003
        return mat

    # plt.imshow(b_func(*[None]*3))
    # plt.show()

    sol.b = b_func

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        dw = sol.dw  # random variable
        t = sol.t

        res['time'].append(t)
        res['p_load'].append(ps.loads['DynamicLoadFiltered'].p(x, v))
        res['g_load'].append(ps.loads['DynamicLoadFiltered'].g_load(x, v).copy())
        res['gen_angle'].append(ps.gen['GEN'].angle(x, v).copy())
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['dw'].append(sol.dw)
        res['freq'].append(((ps.gen['GEN'].speed(x, v).copy()*50)/(2*np.pi))+49.9)


### Frekvens fra taasjuusdata ###
    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/Taajuusdata2024-05-01.csv')
    freq1 = data['Value'][156800:162800]
    time1 = []
    for i in range(len(freq1)):
        time1.append(i / 10)
###Frekvens fra taasjuusdata ###

### Frekvens fra peltondata ###
    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_2_pelton.csv')
    freq2 = data['Frequency [Hz*4]'][2:30000]/4
    time2 = data['Time [s]'][2:30000]
###Frekvens fra peltondata ###
    ###Autocorrelation###

    avg_freq = [np.mean(freq) for freq in res['freq']]
    print(len(avg_freq), len(freq2))


    #CompareAutocorrelations(freq2, avg_freq, 1)



    # fig, ax = plt.subplots(4, sharex=True)
    # ax[0].plot(res['time'], res['gen_speed'])
    # ax[0].set_ylabel('Gen speed')
    # ax[0].set_xlabel('Time [s]')
    # ax[1].plot(res['time'], sol.dt*np.cumsum(np.array(res['dw']), axis=0))
    # ax[1].set_ylabel('W')
    # ax[2].plot(res['time'], res['p_load'])
    # ax[2].set_ylabel('P Load [p.u.]')
    plt.figure(figsize=(10, 8))
    plt.plot(res['time'], avg_freq, label='Simulation')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [s]')
    plt.plot(time2, freq2, label='Real data')
    #ax[3].plot(time1, freq1, label='Real data')
    plt.legend()
    plt.show()





    #df = pd.DataFrame(fineData)
    #df.to_excel('fineData.xlsx', index=False)

    gen_angle_np = np.array(res['gen_angle']).T
    gen_speed_np = np.array(res['gen_speed']).T

    angle_speed_mat = np.vstack([gen_angle_np, gen_speed_np])

    cov_mat = np.cov(angle_speed_mat)
    cov_mat_angle = np.cov(gen_angle_np[[0, 2], :])
    np.linalg.det(cov_mat_angle)

    # np.linalg.inv()

    # Calculate PSD for real frequency values



    ###From peltonplot###
    f2, Pxx2 = signal.periodogram(freq2, 50)
    Pxx2_dB = 20 * np.log10(abs(Pxx2))
    plt.plot(f2, Pxx2_dB, label='Real Data pelton')



    ###From simulation###
    f, Pxx = signal.periodogram(avg_freq, fs=200)
    offset=1
    Pxx_dB = 20 * np.log10(abs(Pxx))
    plt.plot(f, Pxx_dB, label='Simulated Data')
    plt.xlabel('frequency [Hz]')
    plt.xlim(0,1)
    plt.ylim(-200, 80)
    plt.ylabel('PSD [dB/Hz]')
    plt.legend()
    plt.show()

    ###From taasjuusdata###
    data = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/Taajuusdata2024-05-01.csv')
    freq1 = data['Value'][156800:158100]
    time1 = []
    for i in range(len(freq1)):
        time1.append(i / 10)
    f1, Pxx1 = signal.periodogram(freq1, 10)
    Pxx1_dB = 10 * np.log10(Pxx1+1)
    plt.semilogy(f1, Pxx1, label='Real Data taasjuus')



