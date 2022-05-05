import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties.unumpy import uarray
import uncertainties
import matplotlib
from scipy.fft import fft, fftfreq


files = ['002_EOS_FFT.txt', '002_EOS_FFT2.txt']
savefiles = ['plot1.pdf', 'plot2.pdf']


for n in range(len(files)):
    T, P, Zero, Delay, X, Y, R, theta = np.genfromtxt(files[n], delimiter='\t', unpack=True, usecols= (0,1,2,3,4,5,6,7)) #unpack the txt file with pixel values
                                                                                     #I dont know why it wants to unpack 9 values,but thats the reason for the variable m


    #############################
    #   FFT of Data 
    N  = len(X) # Number of Datapoints in X
    timestep_ = np.zeros(len(T)-1)
    for k in range(N-1):
        timestep_[k] = Delay[k] - Delay[k+1]
    timestep = np.abs(np.sum(timestep_)/(N-1))
    print(timestep)
    FX = fft(X)[0:N//2 +1]
    FDelay = fftfreq(len(X),d=timestep)[0:N//2 +1] #Just use the positiv value of the frequencies
    #############################

    data = np.array([Delay, X])
    Fdata = np.array([FDelay,np.abs(FX)])
    data_name = ['X', 'FX']
    data_name_x = ['Delay / ps', 'Frequency / THz']
    data_name_y = ['X', 'Fourier[X]']
    fig, ax = plt.subplots(len(data_name),1, figsize=(16,8))

    ######################
    #   Plotting
    ######################

    plot_data = [data, Fdata]
    for i in range(len(data_name)):
        ax[i].plot(plot_data[i][0], plot_data[i][1], label=data_name[i])
        ax[i].grid()
        ax[i].set_xlabel(data_name_x[i])
        ax[i].set_ylabel(data_name_y[i])
        ax[i].set_title(data_name[i])
        # ax[i].legend()

    plt.tight_layout()
    plt.savefig(savefiles[n])