import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf
import os




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
    FreqData.head(15)
    plot_acf(FreqData)
    plt.show()

def CompareAutocorrelations(FreqData1, FreqData2, myLag, label1=None, label2=None):
    if type(FreqData1)!=pd.Series:
        FreqData1 = pd.Series(FreqData1)
    if type(FreqData2)!=pd.Series:
        FreqData2 = pd.Series(FreqData2)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    pd.plotting.lag_plot(FreqData1, lag=myLag, ax=ax1)
    pd.plotting.lag_plot(FreqData2, lag=myLag, ax=ax2)
    print("Length of frequency data 1 ", label1 ,":", len(FreqData1),"\nLength of frequency data 2 ",label2, ":", len(FreqData2))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    plot_acf(FreqData1, lags=1000, ax=ax1, title=label1, color='blue')
    plot_acf(FreqData2, lags=1000, ax=ax2, title=label2, color='red')

    plt.tight_layout()
    plt.show()

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




