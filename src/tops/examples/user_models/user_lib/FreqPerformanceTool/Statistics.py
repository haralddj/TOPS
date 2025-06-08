import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import os
import concurrent.futures

###Statistical functions used for stochastic analysis of frequency data###


def frequency_from_excel_pelton(filename: str, samples: int, mytype: str = None):
    data = pd.read_csv(filename)
    freq = (data['Frequency [Hz*4]'][2:samples] / 4)
    time = (data['Time [s]'][2:samples])
    if mytype is None:
        freq = (data['Frequency [Hz*4]'][2:samples]/4)
        time = (data['Time [s]'][2:samples])
    elif mytype == 'list':
        freq = list(data['Frequency [Hz*4]'][2:samples]/4)
        time = list(data['Time [s]'][2:samples])
    else:
        print("Invalid type, data returned as pandas dataframe")

    return time, freq


def autocorrelating(FreqData, myLag):
    if type(FreqData)!=pd.Series:
        FreqData = pd.Series(FreqData)
    pd.plotting.lag_plot(FreqData, lag=myLag)
    plot_acf(FreqData)
    plt.show()

def CompareAutocorrelations(SimFreqData, RealFreqData, myLag=12500, timestepSim=0.02, timestepReal=0.02,label1=None, label2=None):
    if type(RealFreqData)!=pd.Series:
        RealFreqData = pd.Series(RealFreqData)
    if type(SimFreqData)!=pd.Series:
        SimFreqData = pd.Series(SimFreqData)
    print("Length of frequency data 1 ", label1 ,":", len(RealFreqData),"\nLength of frequency data 2 ",label2, ":", len(SimFreqData))
    acf_val_freq1=acf(SimFreqData, nlags=myLag)
    acf_val_freq2=acf(RealFreqData, nlags=myLag)
    lagsSim=np.arange(len(acf_val_freq1))*timestepSim
    lagsReal=np.arange(len(acf_val_freq2))*timestepReal
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.plot(lagsSim, acf_val_freq1, color='blue', label=label1)
    ax.plot(lagsReal, acf_val_freq2, color='red', label=label2)
    ax.set_xlabel('Time [s]')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation Comparison')
    ax.legend()
    plt.tight_layout()
    plt.show()

def getAutocorrelationData(FreqData, myLag, label=None):
    if type(FreqData)!=pd.Series:
        RealFreqData = pd.Series(FreqData)
    acf_values=acf(FreqData, nlags=myLag)
    lags=np.arange(len(acf_values))
    return lags, acf_values



def gatherFreqData(folderDirectory: str, samples: int):
    all_data = []
    min_length = float('inf')  # Track the smallest array length

    for filename in os.listdir(folderDirectory):
        if filename.endswith('.csv'):
            time, freq = frequency_from_excel_pelton(os.path.join(folderDirectory, filename), samples)

            time, freq = np.array(time), np.array(freq)

            # Update the minimum length
            min_length = min(min_length, len(time), len(freq))

            if not all_data:
                all_data.append(time)
            all_data.append(freq)

    # Slice all lists to match the smallest length
    all_data = [arr[:min_length] for arr in all_data]

    # Convert to NumPy array
    stacked_data = np.column_stack(all_data)

    print(stacked_data)
    return stacked_data

def averageFrequencyData(stackedData):
    return np.mean(stackedData[:, 1:], axis=1)

def plot_normalDistribution(FreqData, ax=None, title=None, label=None, color=None, zorder=None, linewidth=None, alpha=None):

    if type(FreqData) == str:
        _,FreqData = frequency_from_excel_pelton(FreqData, -1)
    if type(FreqData) != pd.Series:
        FreqData = pd.Series(FreqData)
    if ax is None:
        FreqData.plot(kind='kde', title=title, zorder=zorder, label=label, color=color, linewidth=linewidth, alpha=alpha)

        plt.show()
    else:
        plot_label = label if label is not None else '_nolegend_'
        FreqData.plot(kind='kde', ax=ax, label=plot_label, zorder=zorder, color=color, linewidth=linewidth, alpha=alpha)



def EulerMaryama(theta, mu, sigma, dt, last_value):
    dW = np.random.normal(0, np.sqrt(dt))
    return last_value+theta*(mu-last_value)*dt+sigma*dW

def simulate_terminal(theta, mu, sigma, dt, T):
    """
    Simulate one OU trajectory from X(0)=mu up to time T,
    return only X_T.
    """
    x = mu
    n_steps = int(T / dt)
    for _ in range(n_steps):
        x = EulerMaryama(theta, mu, sigma, dt, x)
    return x

def estimate_terminal_variance(theta, mu, sigma, dt, T, n_reps=1000):
    """
    Run n_reps independent simulations, collect X_T from each,
    and compute mean & variance of those end‐points.
    """
    with concurrent.futures.ProcessPoolExecutor() as exec:
        # spawn n_reps tasks that all call simulate_terminal(...)
        futures = [exec.submit(simulate_terminal, theta, mu, sigma, dt, T)
                   for _ in range(n_reps)]
        terminal_values = [f.result() for f in futures]

    mean_T = np.mean(terminal_values)
    var_T  = np.var(terminal_values, ddof=1)
    return mean_T, var_T, terminal_values

if __name__ == "__main__":
    #Testing Euler-maryama scheme for OU process
    theta, mu, sigma = 0.1, 967.0, 10.0
    dt, T            = 0.02, 2000.0
    mean_T, var_T, _ = estimate_terminal_variance(theta, mu, sigma, dt, T, n_reps=1000)

    print(f"Empirical mean of X_T:     {mean_T:.4f}")
    print(f"Empirical var of X_T:      {var_T:.4f}")
    print(f"Theoretical Var(X_T):     {sigma**2/(2*theta)*(1-np.exp(-2*theta*T)):.4f}")




def MakeEulerMaryamaList(theta, mu, sigma, dt, T):
    list1 = [mu]
    for i in range(int(T /dt)):
        last_value = list1[-1]
        list1.append(EulerMaryama(theta, mu, sigma, dt, last_value))

    LongtermCalcVar = (sigma ** 2) / (2 * theta)
    LongTermCalcVarTimeDependent = ((sigma ** 2) / (2 * theta)) * (1 - np.exp(-2 * theta * T))
    return list1, np.mean(list1), np.var(list1, ddof=1), LongtermCalcVar, LongTermCalcVarTimeDependent




def MakeEulerMaryama(theta, mu, sigma, dt, Y_init, T):
    N = int(T / dt)
    points = [Y_init]
    pointsRandom = [0]

    for i in range(1, N):
        dW = np.random.normal(0, np.sqrt(dt))
        y = points[-1]
        rand=theta * (mu-y) * dt + sigma * dW
        pointsRandom.append(rand)
        y_new = y * np.exp(-theta * dt) + mu * (1 - np.exp(-theta * dt)) + sigma * np.sqrt((1 - np.exp(-2 * theta * dt)) / (2 * theta)) * np.random.normal(0, 1)

        points.append(y_new)
    print()
    print("Mean: ", np.mean(points))
    print("Simulation Variance: ", np.var(np.array(points)))
    LongtermCalcVar = (sigma ** 2) / (2 * theta)
    LongTermCalcVarTimeDependent = (sigma ** 2) / (2 * theta) * (1 - np.exp(-2 * theta * (T)))
    print("Long term variance: ", LongtermCalcVar)
    print("Long term variance time dependent: ", LongTermCalcVarTimeDependent)
    #Calculating the shortest time needed to use this theta, meaning when the variance is 0.95 times the long term variance
    shortestTime=-np.log(0.01)/-(2*theta)
    print("Shortest simulation time needed to use this theta:", shortestTime)
    tt=np.linspace(0, T, N)

    #Plotting the data
    plt.plot(tt, points)
    plt.plot(tt, pointsRandom)
    plt.xlabel('Points')
    plt.ylabel('Value')
    plt.title('Euler Maryama Simulation')
    plt.show()

    return points, pointsRandom








