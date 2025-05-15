import numpy as np
import examples.user_models.user_lib.MyTools.Functions as fc
import examples.user_models.user_lib.ProsjektoppgaveMaster.min_k2a as model_data


def tuneDynLoadModel(model, filepath, columnname):
    timevalues, freqvalues = fc.GetFreqData(filepath, columnname)
    time_freq_dict = dict(zip(timevalues, freqvalues))
    simtime, simfreq = fc.runDynSimulation(model, 25, 4e-2, 2.5, 0, 0.1, 5, 0.1, 7.5, 0.07,  0.9,)
    print(simfreq)
    print(len(simtime), len(simfreq))
    H_lim=[1, 10]
    D_lim=[0, 0]
    Tf_lim=[0.05, 0.1]
    Tr_lim=[5, 30]
    Tg_lim=[0.2, 1.0]
    Tw_lim=[1,10]
    R_lim=[0.05, 0.1]
    r_lim=[0.5, 2]
    h=2.5
    d=0
    Tf=0.1
    Tr=5
    Tg=0.1
    Tw=7.5
    R=0.07
    r=0.9
    deviation=0
    for i in range(0, min(len(simtime), len(simfreq))-1):
        print(i)
        current_t = simtime[i]
        #closest_time = min(time_freq_dict.keys(), key=lambda k: abs(k - current_t))
        deviation += np.sqrt((simfreq[i] - freqvalues[i]) ** 2)
    LowestDev=1e9
    iteration=0
    currentValues = [h, d, Tf, Tr, Tg, Tw, R, r]
    while LowestDev>=10 and iteration<=200:
        h=np.random.choice(np.arange(H_lim[0], H_lim[1], 1))
        #d=np.random.choice(np.arange(D_lim[0], D_lim[1], 1))
        Tf=np.random.choice(np.arange(Tf_lim[0], Tf_lim[1], 0.01))
        Tr=np.random.choice(np.arange(Tr_lim[0], Tr_lim[1], 1))
        Tg=np.random.choice(np.arange(Tg_lim[0], Tg_lim[1], 0.1))
        Tw=np.random.choice(np.arange(Tw_lim[0], Tw_lim[1], 1))
        R=np.random.choice(np.arange(R_lim[0], R_lim[1], 0.01))
        r=np.random.choice(np.arange(r_lim[0], r_lim[1], 0.1))
        simtime, freqvalues = fc.runDynSimulation(model, 20, 5e-2, h, d, Tf, Tr, Tg, Tw, R, r)
        for i in range(0, min(len(simtime), len(simfreq))-1):
            current_t=simtime[i]
            closest_time = min(time_freq_dict.keys(), key=lambda k: abs(k - current_t))
            deviation += np.sqrt((simfreq[i] - time_freq_dict[closest_time]) ** 2)
        if deviation < LowestDev:
            LowestDev = deviation
            currentValues = [h, d, Tf, Tr, Tg, Tw, R, r]
        iteration+=1
    print(currentValues, LowestDev)


tuneDynLoadModel(model_data,'C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Outage-anonymous (1)/Outage-anonymous.xlsx', 'FI south:Frequency' )












