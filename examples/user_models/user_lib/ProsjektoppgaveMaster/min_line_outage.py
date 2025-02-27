import sys
from collections import defaultdict
import matplotlib.pyplot as plt
import time
import tops.dynamic as dps
import tops.solvers as dps_sol
import importlib
importlib.reload(dps)
import numpy as np
import examples.user_models.user_lib.MyTools.Functions as fc

if __name__ == '__main__':

    # Load model
    import examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data
    importlib.reload(model_data)
    model = model_data.load()

    # Power system model
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print(max(abs(ps.state_derivatives(0, ps.x_0, ps.v_0))))

    t_end = 25
    x_0 = ps.x_0.copy()

    # Solver
    sol = dps_sol.ModifiedEulerDAE(ps.state_derivatives, ps.solve_algebraic, 0, x_0, t_end, max_step=4e-3)

    # Initialize simulation
    t = 0
    res = defaultdict(list)
    t_0 = time.time()

    event_flag = True

    # Run simulation
    while t < t_end:
        sys.stdout.write("\r%d%%" % (t/(t_end)*100))

        # Short circuit
        if 1<t and event_flag:
            event_flag = False
            ps.lines['Line'].event(ps, ps.lines['Line'].par['name'][-1], 'disconnect')

        # Simulate next step
        result = sol.step()
        x = sol.y
        v = sol.v
        t = sol.t

        dx = ps.ode_fun(0, ps.x_0)

        # Store result
        res['t'].append(t)
        res['gen_speed'].append(ps.gen['GEN'].speed(x, v).copy())
        res['freq'].append((ps.gen['GEN'].speed(x, v).copy()*50/(2*np.pi))+50.0)

    print('Simulation completed in {:.2f} seconds.'.format(time.time() - t_0))

avg_freq = [np.mean([f for i, f in enumerate(freq) if i != 2]) for freq in res['freq']]
avg_speed = [np.mean([s for i, s in enumerate(speed) if i != 2]) for speed in res['gen_speed']]

timevalues, frequency_data = fc.PlotFreqData(
    'C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous (1)/Outage-anonymous.xlsx',
    'FI south:Frequency')

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.plot(timevalues, frequency_data, label='Measured frequency')
ax1.set_ylabel('Measured frequency')
ax1.legend()

ax2.plot(res['t'], avg_freq, label='Avg. freq.')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel('Avg. freq.')
ax2.legend()

for i, gen_name in enumerate(ps.gen['GEN'].par['name']):
    ax3.plot(res['t'], [speed[i] for speed in res['gen_speed']], label=f'Gen. speed {gen_name}')
ax3.set_xlabel('Time [s]')
ax3.set_ylabel('Gen. speed')
ax3.legend()
plt.show()

