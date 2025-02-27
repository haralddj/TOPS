
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
        return ((p+1j*q)*(v_real-1j*v_imag)/((abs(v_real+1j*v_imag))**2))


    def I_inj(self, x, v):
        i_n= self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'][self.bus_idx_red['terminal']])
        return self.i_inj(x, v)*i_n

    def input_list(self):
        return ['P_setp', 'Q_setp']

    def voltageReal(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def voltageImag(self, x, v):
        return v[self.bus_idx_red['terminal']]

    def p(self, x, v):
        return np.sqrt(3)*abs(v[self.bus_idx_red['terminal']])*self.i_inj(x, v).real

    def q(self, x, v):
        return np.sqrt(3)*abs(v[self.bus_idx_red['terminal']])*self.i_inj(x, v).imag


    def P(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']])*v_n
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'][self.bus_idx_red['terminal']])
        I_real = i_n*self.i_inj(x, v).real
        return np.sqrt(3)*V*I_real

    def Q(self, x, v):
        v_n = self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]
        V = abs(v[self.bus_idx_red['terminal']]) * v_n
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'][self.bus_idx_red['terminal']])
        I_imag = i_n * self.i_inj(x, v).imag
        return np.sqrt(3) * V * I_imag

    def load_flow_pq(self):
        return self.bus_idx['terminal'], self.par['P_setp'], self.par['Q_setp']

    def init_from_load_flow(self, x_0, v_0, S):
        self._input_values['P_setp'] = self.par['P_setp']
        self._input_values['Q_setp'] = self.par['Q_setp']

        v_n= self.sys_par['bus_v_n'][self.bus_idx_red['terminal']]

        input_v0 = v_0[self.bus_idx_red['terminal']]
        P_0 = self.par['P_setp']
        Q_0 = self.par['Q_setp']


        self.lag_P.initialize(x_0, v_0, P_0)
        self.lag_Q.initialize(x_0, v_0, Q_0)

        self.voltageLagReal.initialize(x_0, v_0, input_v0.real)
        self.voltageLagImag.initialize(x_0, v_0, input_v0.imag)

    def current_injections(self, x, v):
        i_n = self.sys_par['s_n'] / (np.sqrt(3) * self.sys_par['bus_v_n'])
        return self.bus_idx_red['terminal'], self.i_inj(x,v)