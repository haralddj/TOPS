import numpy as np
from tops.dyn_models.utils import DAEModel
from tops.dyn_models.blocks import TimeConstant
from tops.dyn_models.blocks import *
from tops.utility_functions import lookup_strings



class Load(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        print(data)
        self.n_units = len(data)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def init_from_load_flow(self, x_0, v_0, S):
        self.v_0 = v_0[self.bus_idx['terminal']]
        s_load = (self.par['P'] + 1j * self.par['Q']) / self.sys_par['s_n']
        z_load = np.conj(abs(self.v_0) ** 2 / s_load)
        self.y_load = 1/z_load

        V_n = self.sys_par['bus_v_n'][self.bus_idx['terminal']]
        self.I_n = self.sys_par['s_n']/(np.sqrt(3)*V_n)

    def dyn_const_adm(self):
        return self.y_load, (self.bus_idx_red['terminal'],)*2

    def i(self, x, v):
        return v[self.bus_idx_red['terminal']]*self.y_load
    
    def I(self, x, v):
        return self.i(x, v)*self.I_n
    
    def s(self, x, v):
        return v[self.bus_idx_red['terminal']]*np.conj(self.i(x, v))

    def p(self, x, v):
        # p.u. system base
        return self.s(x, v).real

    def q(self, x, v):
        # p.u. system base
        return self.s(x, v).imag
    
    def P(self, x, v):
        # MW
        return self.s(x, v).real*self.sys_par['s_n']

    def Q(self, x, v):
        # MVA
        return self.s(x, v).imag*self.sys_par['s_n']


class DynamicLoad(DAEModel):
    def __init__(self, data, sys_par, **kwargs):
        super().__init__(data, sys_par, **kwargs)
        self.data = data
        self.par = data
        self.n_units = len(data)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.sys_par = sys_par  # {'s_n': 0, 'f_n': 50, 'bus_v_n': None}
        print(sys_par)


    def input_list(self):
        return ['g_setp', 'b_setp']
    
    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def reduced_system(self):
        return self.par['bus']

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P'], self.par['Q']

    def init_from_load_flow(self, x_0, v_0, S):
        self.v_0 = v_0[self.bus_idx['terminal']]
        s_load = (self.par['P'] + 1j * self.par['Q']) / self.sys_par['s_n']
        z_load = np.conj(abs(self.v_0) ** 2 / s_load)
        y_load = 1/z_load
        self._input_values['g_setp'] = y_load.real
        self._input_values['b_setp'] = y_load.imag

        V_n = self.sys_par['bus_v_n'][self.bus_idx['terminal']]
        self.I_n = self.sys_par['s_n']/(np.sqrt(3)*V_n)

    def g_load(self, x, v):
        return self.g_setp(x, v)

    def b_load(self, x, v):
        return self.b_setp(x, v)
    def event(self, ps, load_index, event_name, value):
        if event_name == 'set_g':
            ps.loads['DynamicLoad'].set_input('g_setp', value, load_index)
        elif event_name == 'set_b':
            ps.loads['DynamicLoad'].set_input('b_setp', value, load_index)
    def y_load(self, x, v):
        return self.g_load(x, v) + 1j*self.b_load(x, v)

    def dyn_var_adm(self, x, v):
        return self.y_load(x, v), (self.bus_idx_red['terminal'],)*2

    def i(self, x, v):
        return v[self.bus_idx_red['terminal']]*self.y_load(x, v)
    
    def I(self, x, v):
        return self.i(x, v)*self.I_n
    
    def s(self, x, v):
        return v[self.bus_idx_red['terminal']]*np.conj(self.i(x, v))

    def p(self, x, v):
        # p.u. system base
        return self.s(x, v).real

    def q(self, x, v):
        # p.u. system base
        return self.s(x, v).imag
    
    def P(self, x, v):
        # MW
        return self.s(x, v).real*self.sys_par['s_n']

    def Q(self, x, v):
        # MVA
        return self.s(x, v).imag*self.sys_par['s_n']


class DynamicLoadFiltered(DynamicLoad):
    """Dynamic load where the input is filtered using a low pass filter. 
    
    The load is an admittance which is determined by the output of two low pass filters,
    one for G (conductance) and one for B (susceptance).
    """

    def add_blocks(self):
        p = self.par
        self.lpf_g = TimeConstant(T=p['T_g'])
        self.lpf_g.input = lambda x, v: self.g_setp(x, v)

        self.lpf_b = TimeConstant(T=p['T_b'])
        self.lpf_b.input = lambda x, v: self.b_setp(x, v)

    def g_load(self, x, v):
        return self.lpf_g.output(x, v)

    def b_load(self, x, v):
        return self.lpf_b.output(x, v)

    def init_from_load_flow(self, x_0, v_0, S):
        self.v_0 = v_0[self.bus_idx['terminal']]
        s_load = (self.par['P'] + 1j * self.par['Q']) / self.sys_par['s_n']
        z_load = np.conj(abs(self.v_0) ** 2 / s_load)
        y_load = 1/z_load
        self._input_values['g_setp'] = y_load.real
        self._input_values['b_setp'] = y_load.imag

        self.lpf_g.initialize(x_0, v_0, y_load.real)
        self.lpf_b.initialize(x_0, v_0, y_load.imag)

        V_n = self.sys_par['bus_v_n'][self.bus_idx['terminal']]
        self.I_n = self.sys_par['s_n']/(np.sqrt(3)*V_n)


class LoadAsCurrent(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        par = self.par
        self.P_ref= lambda x, v: self.P_setp(x, v)
        self.Q_ref= lambda x, v: self.Q_setp(x, v)

        self.lag_P= TimeConstant(T=par['T_i'])
        self.lag_Q= TimeConstant(T=par['T_i'])
        self.lag_P.input= lambda x, v: self.P_ref(x,v)
        self.lag_Q.input= lambda x, v: self.Q_ref(x,v)

        self.P_block=self.lag_P.output
        self.Q_block=self.lag_Q.output


        self.v_real_in=lambda x, v: v[self.bus_idx_red['terminal']].real
        self.v_imag_in=lambda x, v: v[self.bus_idx_red['terminal']].imag
        self.voltageLagReal= TimeConstant(T=par['T_v'])
        self.voltageLagImag= TimeConstant(T=par['T_v'])
        self.voltageLagReal.input= lambda x, v: self.v_real_in(x,v)
        self.voltageLagImag.input= lambda x, v: self.v_imag_in(x,v)

        self.v_real = self.voltageLagReal.output
        self.v_imag = self.voltageLagImag.output



    def i_inj(self, x, v):

        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        p=self.P_block(x,v)/self.sys_par['s_n']
        q=self.Q_block(x,v)/self.sys_par['s_n']

        v_real=self.v_real(x,v)
        v_imag=self.v_imag(x,v)

        return -np.conj((p+1j*q)/(v_real+1j*v_imag))
        #return ((p-1j*q)*(v_real+1j*v_imag)/((abs(v_real+1j*v_imag))**2))/np.sqrt(3)


    def I_inj(self, x, v):
        i_n= self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'][self.bus_idx_red['terminal']])
        return self.i_inj(x, v)*i_n

    def input_list(self):
        return ['P_setp', 'Q_setp']

    def voltageReal(self, x, v):

        return v[self.bus_idx_red['terminal']].real

    def voltageImag(self, x, v):

        return v[self.bus_idx_red['terminal']].imag

    def p(self, x, v):
        return np.sqrt(3)*abs(v[self.bus_idx_red['terminal']])*self.i_inj(x, v).real

    def q(self, x, v):
        return np.sqrt(3)*abs(v[self.bus_idx_red['terminal']])*self.i_inj(x, v).imag


    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'][self.bus_idx_red['terminal']])
        I_real = i_n*abs(self.i_inj(x,v))*np.cos(np.angle(self.i_inj(x,v)))
        return np.sqrt(3)*V*I_real

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']]) * v_n
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'][self.bus_idx_red['terminal']])
        I_imag = i_n*abs(self.i_inj(x,v))*np.sin(np.angle(self.i_inj(x,v)))
        return np.sqrt(3) * V * I_imag

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P_setp'], self.par['Q_setp']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['P_setp'] = self.par['P_setp']
        self._input_values['Q_setp'] = self.par['Q_setp']

        v_n= self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]

        input_v0_real = v_0[self.bus_idx_red['terminal']].real
        input_v0_imag = v_0[self.bus_idx_red['terminal']].imag
        print("Input voltage real init:",input_v0_real)
        print("Input voltage imag init:",input_v0_imag)
        P_0 = self.par['P_setp']
        Q_0 = self.par['Q_setp']


        self.lag_P.initialize(x_0, v_0, P_0)
        self.lag_Q.initialize(x_0, v_0, Q_0)

        self.voltageLagReal.initialize(x_0, v_0, input_v0_real)
        self.voltageLagImag.initialize(x_0, v_0, input_v0_imag)

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        return self.bus_idx_red['terminal'], self.i_inj(x,v)





