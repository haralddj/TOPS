import matplotlib.pyplot as plt
import numpy as np
from tops.examples.user_models.user_lib.MyTools.Statistics import plot_normalDistribution, plotFFT, CompareAutocorrelations, EulerMaryama
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import acf
import pandas as pd
from scipy.signal import welch
import wesanderson as ws
import os

colors=ws.film_palette('Darjeeling limited')


whiteNoiseSignal=list(np.random.normal(0, 1, 10000))

whiteNoiseSignalRepeating=list(np.random.normal(0, 1, 100000))
for i in range(9999):
    if i%100==0:
        whiteNoiseSignalRepeating[i]=50


plt.figure()
plt.plot(acf(whiteNoiseSignalRepeating, nlags=10000), color=colors[0], label='White Noise Signal')
plt.plot(acf(whiteNoiseSignal, nlags=10000), color=colors[1], label='White Noise Signal Repeating')
plt.legend()
plt.show()




# def makeFFT(timeseries, timestep):
#     N = len(timeseries)  # Number of samples
#     T = timestep  # Sampling time (timestep from simulation)
#     fs = 1 / T  # Sampling frequency
#     print(fs)
#     fstep=fs/N   #Frequency interval
#     t=np.linspace(0, (N-1)*timestep, N)

#     # Compute FFT
#     fft_values = np.fft.fft(timeseries)
#     f=np.fft.fftfreq(N, d=timestep)
#     fft_mag=np.abs(fft_values)/N  #Magnitude of FFT

#     f_plot= f[0:int(N/2+1)]
#     fft_mag_plot=2*fft_mag[0:int(N/2+1)]  #Only plot the first half of the spectrum

#     # Fix Nyquist component (only if N is even)
#     if N % 2 == 0:
#         fft_mag_plot[-1] /= 2  # Undo doubling for Nyquist frequency

#     # # plot
#     # fig, [ax1, ax2] = plt.subplots(nrows=2, ncols=1)
#     # ax1.plot(t, timeseries)
#     # ax1.set_xlabel('Time (s)')
#     # ax1.set_ylabel('Amplitude')
#     #
#     # ax2.plot(f_plot, fft_mag_plot)
#     # ax2.set_xlabel('Frequency (Hz)')
#     # ax2.set_ylabel('Magnitude')
#     # plt.legend()
#     # plt.show()
#     return f_plot, fft_mag_plot

# def makeWelchFFT(timeseries, timestep, nperseg=20000, scaling='spectrum'):
#     f_plot, fft_magplot=makeFFT(timeseries, timestep)
#     N=len(timeseries)
#     T=timestep
#     fs=1/T
#     f, Pxx = welch(timeseries, fs, nperseg=nperseg, scaling=scaling, noverlap=nperseg//2)
#     Pxx = np.sqrt(Pxx)* np.sqrt(2)            #going from rms V**2 to just V (magnitude)


#     #delta_f=f[1]-f[0]         #frequency resolution
#     print("Datapoints per segment Welch:",nperseg)
#     plt.semilogx(f, Pxx, label="Welch")
#     plt.semilogx(f_plot, fft_magplot, label="FFT")
#     plt.xlabel('frequency [Hz]')
#     plt.ylabel('Magnitude')
#     plt.legend()
#     plt.show()
#     return f, Pxx


# figures_folder = 'C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/Figures'
# fft_check_folder = os.path.join(figures_folder, 'FFT_check')
# os.makedirs(fft_check_folder, exist_ok=True)


# omega=2*np.pi*1
# t=0
# t_end=100
# signal=[]
# times=[]
# while t<t_end:
#     signal.append(3*np.sin(0.01*omega*t)+3*np.sin(0.5*omega*t) + 2*np.sin(1.0*omega*t) + 1*np.sin(2.5*omega*t) + 4*np.sin(3.0*omega*t))
#     times.append(t)
#     t+=0.02

# f_fft, fft_mag=makeFFT(signal, 0.02)

# # Define the paths for the saved plots
# signal_plot_path = os.path.join(fft_check_folder, 'multi_tone_signal.pdf')
# fft_plot_path = os.path.join(fft_check_folder, 'fft_multi_tone_signal.pdf')


# #Plotting the signal
# plt.plot(times, signal, color=colors[2], label='Multi-tone Sine Signal')
# plt.title('Multi-tone Sine Signal')
# plt.xlabel('Time [s]')
# plt.ylabel('Amplitude')
# plt.legend()
# #plt.savefig(signal_plot_path, format='pdf', bbox_inches='tight')  # Save the signal plot

# plt.show()

# #Plotting the FFT and the welch of the signal

# #f_welch, Pxx=makeWelchFFT(signal, 0.02, nperseg=200)
# plt.plot(f_fft, fft_mag, color=colors[1], label='FFT of Multi-tone Sine Signal')
# #plt.plot(f_welch, Pxx)
# plt.title('FFT of Multi-tone Sine Signal')
# plt.xlabel('Frequency [Hz]')
# plt.ylabel('Magnitude')
# plt.xlim(-0.1,4)
# plt.legend()
# plt.savefig(fft_plot_path, format='pdf', bbox_inches='tight')  # Save the FFT plot
# plt.show()

# data1 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_1.csv')
# data2 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_2_pelton.csv')
# data3 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_3.csv')
# data4 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_4.csv')
# data5 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_5.csv')
# data6 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_6.csv')
# data7 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_7.csv')
# data8 = pd.read_csv('C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/2024-05/PeltonData/data_set_8.csv')

# #Plotting the acf of all the datasets in the same plot for 200 seconds (10000 lags)
# print(data1)
# print(data1['Frequency [Hz*4]'])
# acf_val_freq1=acf(data1['Frequency [Hz*4]']/4, nlags=10000)
# acf_val_freq2=acf(data2['Frequency [Hz*4]']/4, nlags=10000)
# acf_val_freq3=acf(data3['Frequency [Hz*4]']/4, nlags=10000)
# acf_val_freq4=acf(data4['Frequency [Hz*4]']/4, nlags=10000)
# acf_val_freq5=acf(data5['Frequency [Hz*4]']/4, nlags=10000)
# acf_val_freq6=acf(data6['Frequency [Hz*4]']/4, nlags=10000)
# acf_val_freq7=acf(data7['Frequency [Hz*4]']/4, nlags=10000)
# acf_val_freq8=acf(data8['Frequency [Hz*4]']/4, nlags=10000)

# import os

# # Define the path for the new folder
# figures_folder = 'C:/Users/haral/PycharmProjects/ProsjektOppgaveTOPS/figures'
# new_folder = os.path.join(figures_folder, 'autocorrelation_plots')

# # Create the new folder if it doesn't exist
# os.makedirs(new_folder, exist_ok=True)

# # Define the path for the PDF file
# pdf_path = os.path.join(new_folder, 'autocorrelation_plot.pdf')

# # Plot the figure
# timevalues = np.arange(len(acf_val_freq1)) * 0.02  # Seconds

# plt.plot(timevalues, acf_val_freq1, color='blue', alpha=0.2)
# plt.plot(timevalues, acf_val_freq2, color='blue', alpha=0.2)
# plt.plot(timevalues, acf_val_freq3, color='blue', alpha=0.2)
# plt.plot(timevalues, acf_val_freq4, color='blue', alpha=0.2)
# plt.plot(timevalues, acf_val_freq5, color='blue', alpha=0.2)
# plt.plot(timevalues, acf_val_freq6, color='blue', alpha=0.2)
# plt.plot(timevalues, acf_val_freq7, color='blue', alpha=0.2)
# plt.plot(timevalues, acf_val_freq8, color='blue', label='Real Frequency Data Autocorrelation', alpha=0.2)

# plt.plot(timevalues, np.exp(-1 * timevalues), label=r'$e^{-1t}$', linestyle='--')
# plt.plot(timevalues, np.exp(-0.1 * timevalues), label=r'$e^{-0.1t}$', linestyle='--')
# plt.plot(timevalues, np.exp(-0.05 * timevalues), label=r'$e^{-0.05t}$', linestyle='--')
# plt.plot(timevalues, np.exp(-0.025 * timevalues), label=r'$e^{-0.025t}$', linestyle='--')
# plt.plot(timevalues, np.exp(-0.01 * timevalues), label=r'$e^{-0.01t}$', linestyle='--')
# plt.plot(timevalues, np.exp(-0.005 * timevalues), label=r'$e^{-0.0075t}$', linestyle='--')

# plt.legend()

# plt.title('Autocorrelation of all datasets and Exponential Functions')
# plt.xlabel('Time [s]')
# plt.ylabel('Autocorrelation')

# # Save the figure as a PDF
# plt.savefig(pdf_path, format='pdf', bbox_inches='tight')

# # Show the plot
# plt.show()






