from tops.dyn_models.blocks import *
#from .pll import PLL1



''' Attempting to create a load model as a current injection model, inspired by Run_vsc.py. To be used for stochastic modelling'''
class LoadAsCurrent(DAEModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bus_idx = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        self.bus_idx_red = np.array(np.zeros(self.n_units), dtype=[(key, int) for key in self.bus_ref_spec().keys()])
        #self.whiteNoiseP =

    def bus_ref_spec(self):
        return {'terminal': self.par['bus']}

    def add_blocks(self):
        p = self.par

        self.S_ref= lambda x, v: self.P_setp(x, v) + 1j*self.Q_setp(x, v)


        self.dw= self.par['whiteNoiseP']+1j*self.par['whiteNoiseQ']
        self.noiseInput= self.par['K']*self.dw


        self.integratorS=Integrator(n_units=self.n_units)
        self.dS= lambda x, v: (self.S_ref(x,v) - (self.P(x,v)+1j*self.Q(x,v)) + self.noiseInput)*(1/self.par['T_i'])
        self.integratorS.input= self.dS

        self.voltageLag= TimeConstant(T=p['T_v'])
        self.voltageLag.input= lambda x, v: v[self.bus_idx_red['terminal']]


        self.I= lambda x,v: self.integratorS.output*np.conj(self.voltageLag.output)/(abs(self.voltageLag)**2)


    def I_inj(self, x, v):
        return (self.I)

    def input_list(self):
        return ['P_setp', 'Q_setp']

    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*np.real(self.I(x, v))

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        return np.sqrt(3)*V*np.imag(self.I(x, v))

    def load_flow_pq(self):
        return self.bus_idx['terminal'], -self.par['P_setp'], -self.par['Q_setp']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['P_setp'] = self.par['P_setp']
        self._input_values['Q_setp'] = self.par['Q_setp']

        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]

        V_0 = v_0[self.bus_idx_red['terminal']]*v_n

        dw_0=0
        '''I_d_0 = self.par['P_setp']/(abs(V_0)*np.sqrt(3))
        I_q_0 = self.par['Q_setp']/(abs(V_0)*np.sqrt(3))'''
        self.I_0=(self.par['P_setp']+1j*self.par['Q_setp'])()/(abs(V_0)*np.sqrt(3))

        self.integratorS.initialize(x_0, v_0, self.I_0)


    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        # self.P(x, v)
        return self.bus_idx_red['terminal'], self.I_inj(x, v)/i_n[self.bus_idx_red['terminal']]