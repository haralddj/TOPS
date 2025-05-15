import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import os
import concurrent.futures




#Filenames
dataset1='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_1.csv'
dataset2='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_2_pelton.csv'
dataset3='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_3.csv'
dataset4='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_4.csv'
dataset5='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_5.csv'
dataset6='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_6.csv'
dataset7='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_7.csv'
dataset8='C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_8.csv'


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
'''
TestTime, TestFrequency=frequency_from_excel_pelton(dataset1, -1)
plt.plot(TestTime[:2000], TestFrequency[:2000])
plt.xlabel('Time [s]')
plt.ylabel('Frequency [Hz]')
plt.title('Frequency vs Time for the first 2000 values')
plt.show()'''

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
    # fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    # pd.plotting.lag_plot(RealFreqData, lag=myLag, ax=ax1)
    # pd.plotting.lag_plot(SimFreqData, lag=myLag, ax=ax2)
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


#CompareAutocorrelations(TestFrequency[1:], TestFrequency[:-1], 1000)

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

def plot_normalDistribution(FreqData, ax=None, title=None, label=None):

    if type(FreqData) == str:
        _,FreqData = frequency_from_excel_pelton(FreqData, -1)
    if type(FreqData) != pd.Series:
        FreqData = pd.Series(FreqData)
    if ax is None:
        FreqData.plot(kind='kde', title=title)

        plt.show()
    else:
        FreqData.plot(kind='kde', ax=ax, label=label)

#plot_normalDistribution(TestFrequency)
#stackedData=gatherFreqData('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData', -1)

def plotFFT(signal, timestep=5e-3, frequency_range=None):
    N = len(signal)  # Number of samples
    T = timestep  # Sampling time (timestep from simulation)
    fs = 1 / T  # Sampling frequency

    # Compute FFT
    fft_values = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(N, d=T)  # Compute frequency bins

    # Take only the positive half of the spectrum
    half_N = N // 2
    frequencies = frequencies[:half_N]
    amplitude_spectrum = np.abs(fft_values[:half_N])  # Magnitude of FFT

    # Plot amplitude spectrum
    plt.figure(figsize=(10, 5))
    plt.plot(frequencies, amplitude_spectrum)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('Amplitude Spectrum')
    plt.xlim(-1, frequency_range)
    plt.grid()
    plt.show()


'''
fig, ax = plt.subplots()
plot_normalDistribution(dataset1, ax=ax)
plot_normalDistribution(dataset2, ax=ax)
plot_normalDistribution(dataset3, ax=ax)
plot_normalDistribution(dataset4, ax=ax)
plot_normalDistribution(dataset5, ax=ax)
plot_normalDistribution(dataset6, ax=ax)
plot_normalDistribution(dataset7, ax=ax)
plt.show()'''

def EulerMaryama(theta, mu, sigma, dt, last_value):
    dW = np.random.normal(0, np.sqrt(dt))
    return last_value+theta*(mu-last_value)*dt+sigma*dW




def MakeEulerMaryamaList(theta, mu, sigma, dt, T):
    list1 = [mu]
    for i in range(int(T /dt)):
        last_value = list1[-1]
        list1.append(EulerMaryama(theta, mu, sigma, dt, last_value))
    # print("Mean: ", np.mean(list1))
    # print("Simulation Variance: ", np.var(list1))
    LongtermCalcVar = (sigma ** 2) / (2 * theta)
    LongTermCalcVarTimeDependent = ((sigma ** 2) / (2 * theta)) * (1 - np.exp(-2 * theta * T))
    # print("Long term variance: ", LongtermCalcVar)
    # print("Long term variance time dependent: ", LongTermCalcVarTimeDependent)
    # shortestTime=-(np.log(0.01))/(2*theta)
    # print("Shortest simulation time needed to use this theta:", shortestTime)
    # Plotting the data
    # plt.plot(list1)
    # plt.xlabel('Points')
    # plt.ylabel('Value')
    # plt.title('Euler Maryama Simulation')
    # plt.show()
    return list1, np.mean(list1), np.var(list1, ddof=1), LongtermCalcVar, LongTermCalcVarTimeDependent

def single_simulation(theta):
    _, mean, var, Longtermvar, LongtermCalcVar = MakeEulerMaryamaList(theta, 967, 60, 0.01, 2000)
    return mean, var, Longtermvar, LongtermCalcVar


# ✅ Put your main code inside this block
if __name__ == "__main__":
    theta_values = [0.01, 0.1]

    for theta in theta_values:
        print(f"\nRunning simulations for theta = {theta}...")

        means = []
        variances = []

        with concurrent.futures.ProcessPoolExecutor() as executor:
            results = executor.map(single_simulation, [theta] * 1000)

        for mean, var, longterm_var, longterm_time_var in results:
            means.append(mean)
            variances.append(var)

        print("Average Mean: ", np.mean(means))
        print("Average Variance: ", np.mean(variances))
        print("Theoretical Long-term Variance:", longterm_var)
        print("Time-Dependent Theoretical Variance at T=2000:", longterm_time_var)


#MakeEulerMaryamaList(0.1, 0, 10, 0.02, 1000)



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

#MakeEulerMaryama(0.01, 0, 500, 0.02, 0, 10000)







