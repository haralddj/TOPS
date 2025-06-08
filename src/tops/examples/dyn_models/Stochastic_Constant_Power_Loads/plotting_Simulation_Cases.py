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

from tops.examples.user_models.user_lib.FreqPerformanceTool.Statistics import (
    plot_normalDistribution, CompareAutocorrelations, 
    EulerMaryama, getAutocorrelationData, MakeEulerMaryamaList
)
from tops.examples.user_models.user_lib.FreqPerformanceTool.Functions_utility import *
import wesanderson as ws

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1) Folder where all your res_sim_*.parquet files live
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

###AI used to clean up code so that it is more readable for others###
### For plotting results obtained throggh MonteCarlo_stochastic_Simulations.py ###


# 1) Folder where all your res_sim_*.parquet files live
base_path = r"pathToYourParquetFiles"

# 2) Read each file and collect f0 series
f0_list = []
for i in range(50):
    fname = (
        f"res_sim_{i}_"
        "filaNameofspecificcase.parquet"  # Adjust this to your specific case
    )
    full_path = os.path.join(base_path, fname)
    if not os.path.isfile(full_path):
        raise FileNotFoundError(f"Could not find {full_path}")
    df = pd.read_parquet(full_path)
    f0_list.append(df["f0"].values)
    if i == 0:
        t = df["t"].values

# 3) Stack into shape (50, n_timesteps)
f0_arr = np.vstack(f0_list)

# 4) Compute min/max envelope
f0_min = f0_arr.min(axis=0)
f0_max = f0_arr.max(axis=0)

# 5) Count out-of-band points
band_low, band_high = 49.9, 50.1
total_points   = f0_arr.size
outside_mask   = (f0_arr < band_low) | (f0_arr > band_high)
outside_points = outside_mask.sum()
percent_out    = 100 * outside_points / total_points

# 6) Plot
plt.figure()

# a) dashed band limits
plt.axhline(band_low,  color='gray', linestyle='--', alpha=0.5)
plt.axhline(band_high, color='gray', linestyle='--', alpha=0.5)

# b) envelope fill
plt.fill_between(t, f0_min, f0_max, alpha=0.5)

# c) highlight out-of-band regions in dark red
plt.fill_between(t, f0_max, band_high, where=(f0_max > band_high),
                 color='red', alpha=0.7)
plt.fill_between(t, f0_min, band_low,  where=(f0_min < band_low),
                 color='red', alpha=0.7)

# d) envelope boundary lines
plt.plot(t, f0_min, label="Min. Frequency")
plt.plot(t, f0_max, label="Max. Frequency")

# e) legend with percentage out of band
plt.legend(title=f"{percent_out:.2f}% of points\noutside 49.9–50.1 Hz")

plt.xlabel("Time [s]")
plt.ylabel("Frequency [Hz]")
plt.title(r"Range of Average Frequency Across 50 Simulations, Multiple Ratings")
plt.grid(True)
plt.tight_layout()
plt.show()


# plt.figure()
# plt.plot(t_list, p_bus_7_list, label="p_bus_7", color=colors[4])
# plt.show()