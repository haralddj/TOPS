import sys
import time
import importlib
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy

import tops.dynamic as dps
import tops.solvers as dps_sol
from tops.dyn_models.utils import DAEModel

from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf

from tops.examples.user_models.user_lib.MyTools.Statistics import (
    plot_normalDistribution, plotFFT, CompareAutocorrelations, 
    EulerMaryama, getAutocorrelationData, MakeEulerMaryamaList
)
from tops.examples.user_models.user_lib.MyTools.Functions import *
import wesanderson as ws

colors=ws.film_palette('Darjeeling limited')


df = pd.read_parquet(r"C:\Users\haral\PycharmProjects\ProsjektOppgaveTOPS\Figures\SimulationStorageFreqAndLoad\res_sim_0_H_6_R_0.12_r_0.65_Tf_0.05_Tr_5.7_Tg_0.5_Tw_1.1.parquet")

# 2a. Get each column as its own list
t_list       = df["t"].tolist()
p_bus_7_list = df["p_bus_7"].tolist()
f0_list      = df["f0"].tolist()
f1_list      = df["f1"].tolist()
f2_list      = df["f2"].tolist()
f3_list      = df["f3"].tolist()
print(f1_list[:10])

plt.figure()
plt.plot(t_list, f0_list, label="f0", color=colors[0])
plt.plot(t_list, f1_list, label="f1", color=colors[1])
plt.plot(t_list, f2_list, label="f2", color=colors[2])
plt.plot(t_list, f3_list, label="f3", color=colors[3])
plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title("Simulated Frequencies")
plt.legend()
plt.grid()
plt.show()

plt.figure()
plt.plot(t_list, p_bus_7_list, label="p_bus_7", color=colors[4])
plt.show()