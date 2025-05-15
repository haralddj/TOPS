# Synchronous machine connected to passive load

def load():
    return {
        'base_mva': 900,
        'f': 50,

        'buses': [
                ['name',    'V_n'],
                ['B1',      20],
                ['B2',      20],
        ],

        'lines': [
                ['name',    'from_bus',    'to_bus',    'length',   'S_n',  'V_n',  'unit',     'R',    'X',    'B'],
                ['L1-2',    'B1',          'B2',        25,         900,    20,     'PF',       1e-4,   1e-3,   1.75e-3*0],
        ],

        'loads': [
            ['name',    'bus',  'P',    'Q',    'model'],
            ['L1',      'B2',   600.0,    200.0,    'Z'],
        ],

        'generators': {
            'GEN': [
                ['name',    'bus',  'S_n',  'V_n',  'P',    'V',    'H',    'D',    'X_d',      'X_q',  'X_d_t',    'X_q_t',    'X_d_st',   'X_q_st',   'T_d0_t',   'T_q0_t',   'T_d0_st',  'T_q0_st'],
                ['G1',      'B1',   42000,    20,     700,    1,      4.5238,    0,      1.8,        1.7,    0.3,        0.3,        0.2,        0.2,        8.0,        0.6,        0.05,       0.05],
            ],
        },
        'gov': {
            'HYGOV': [
                ['name',  'gen',    'R',    'r',    'T_f',  'T_r',  'T_g',   'A_t',     'T_w', 'q_nl', 'D_turb', 'g_min',     'V_elm',      'g_max', 'P_N'],
                ['HYGOV1', 'G1',    0.14, 0.65,      0.05,    5.7,    0.6,         1,     1.1,   0.01,    0.01,      0,          0.15,        1,      0],
            ]
        },

        'avr': {
            'SEXS': [
                ['name', 'gen', 'K', 'T_a', 'T_b', 'T_e', 'E_min', 'E_max'],
                ['AVR1', 'G1', 25, 2.0, 10.0, 0.1, -10, 10],
            ]
        }, 

        'pss': {
            'STAB1': [ 
                ['name',    'gen',  'K',      'T',      'T_1',  'T_2',  'T_3',  'T_4',  'H_lim'],
                ['PSS1',     'G1',   50,     10,      0.5,      0.5,     0.05,   0.05,   0.03],
            ]}
    }