# TOPS (**T**iny **O**pen **P**ower System **S**imulator)

##########################################################
#################README for TOPS main#####################
##########################################################

**Note**: This repository was previously called DynPSSimPy.


This is a package for performing dynamic power system simulations in Python. The aim is to provide a simple and lightweight tool which is easy to install, run and modify, to be used by researchers and in education. Performance is not the main priority. The only dependencies are numpy, scipy, pandas and matplotlib (the core functionality only uses numpy and scipy).

The package is being developed as part of ongoing research, and thus contains experimental features. Use at your own risk!

Some features:
- Newton-Rhapson power flow
- Dynamic time domain simulation (RMS/phasor approximation)
- Linearization, eigenvalue analysis/modal analysis

# Installation
The package can be installed using pip, as follows:

`pip install tops`

# Citing
If you use this code for your research, please cite [this paper](https://arxiv.org/abs/2101.02937).

# Example notebooks
[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/hallvar-h/TOPS/HEAD?filepath=examples%2Fnotebooks)

# Contact
[Hallvar Haugdal](mailto:hallvhau@gmail.com)

##########################################################
###########README for Stochastic Load Simualtions#########
##########################################################

This branch is an extension of the original TOPS repocitory for simulating and validating stochasticity using constant power loads.
# In "..\src\tops\examples\dyn_models\Stochastic_Constant_Power_Loads" 
all simulator files are found. This entails the power system simulators with
constant power loads along with several ways of comparing frequencies and testing the model units against the frequency performance requirement.
This folder also contains the file used to compare the nonlinear tops system and the linear requirement representation for the same disturbance.

# In "...src\tops\examples\user_models\user_lib\FreqPerformanceTool"
The utility functions used to compare and validate frequency behavior is stored.
The function for simulating an Euler-Maryama scheme, and several functions for statistical analysis are also stored here.

# In "...src\tops\examples\user_models\user_lib\k2aTunedToGrid\k2aTuning.py" 
the k2a model, as tuned to behave as the nordic grid for a given large disturbane is found.

# In "...\storage_simulationData"
a csv file for a large disturbance is stored to be used as an example when running linearSystemSimulation.py

# In "...\HistoricFreqData"
The historic frequency data from the Nordic Grid used throughout this thesis has been stored. 
Outage_anonymous(1) contains the frequency data for a large disturbance (1400MW).
PeltonData contains 8 hour datasets used for stochastic simulation comparisons.
Taajuus2024-05 is data from normal operation with a lower frequency sample rate than PeltonData.

