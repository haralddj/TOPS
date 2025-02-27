import tops.dynamic as dps
from tops.simulator import Simulator
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class ResultKeeper:
    def __init__(self, sim):
        self.t = []
        self.x = []
        self.sim = sim

    def update(self, sim):
        self.t.append(sim.sol.t)
        self.x.append(sim.sol.x.copy())

    def get_dataframe(self):
        df = pd.DataFrame(columns=self.sim.ps.state_desc, data=self.x, index=self.t)
        return df


class Events:
    def __init__(self, sim, data):
        self.data = data
        self.next_event_time = None
        self.next_event_data = None

    def update(self, sim):
        if len(self.data) == 0 and self.next_event_time is None:
            # Could be stopped
            return
        
        if self.next_event_time is None:
            self.next_event_time, self.next_event_data = self.data.pop(0)

        if self.next_event_time <= sim.sol.t:
            if self.next_event_data[0] == 'line':
                sim.ps.lines['Line'].event(sim.ps, self.next_event_data[1], self.next_event_data[2])
            self.next_event_time = None
            if self.next_event_data[0] == 'dynload':
                sim.ps.loads['DynamicLoad'].set_input(self.next_event_data[1], self.next_event_data[2], self.next_event_data[3])
            self.next_event_time = None



if __name__ == '__main__':

    import examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data

    model = model_data.load()
    model['loads'] = {'DynamicLoad': model['loads']}
    ps = dps.PowerSystemModel(model=model)
    ps.init_dyn_sim()

    print(ps.state_desc)

    # ps.ode_fun(0, ps.x0)
    sim = Simulator(ps, dt=5e-3, t_end=20)
    res_keeper = ResultKeeper(sim)
    events = Events(sim, [
        #(1, ('line', 'L6-7', 'disconnect')),
        #(1.2, ('line', 'L6-7', 'connect')),
        (1,  ('dynload', 'g_setp', 2.66, 0)),
        (1,  ('dynload', 'b_setp', -0.3, 0)),
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

    # Calculate the average speed
    df['average_speed'] = df[speed_columns].mean(axis=1)
    df['avg_freq'] = (df['average_speed']*50 / (2 * np.pi)) + 50
    df['avg_freq'].plot()
    time_freq_list = [df.index.tolist(), df['avg_freq'].tolist()]
    plt.show()