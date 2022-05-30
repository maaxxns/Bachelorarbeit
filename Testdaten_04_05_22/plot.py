import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties.unumpy import uarray
import uncertainties
import matplotlib
from scipy.fft import fft, fftfreq


files = ['16_37_35.txt', '16_27_14.txt', '16_21_02.txt', '16_18_12.txt']
savefiles = ['plot16_37_35.pdf', 'plot16_27_14.pdf', 'plot16_21_02.pdf', 'plot16_18_12.pdf']


for n in range(len(files)):
    X, Y, R, theta = np.genfromtxt(files[n], delimiter='\t', unpack=True, usecols= (0,1,2,3), skip_header=1) #unpack the txt file with pixel values
                                                                                     #I dont know why it wants to unpack 9 values,but thats the reason for the variable m
    #############################
    #   FFT of Data 
    N  = len(X) # Number of Datapoints in X
    #timestep_ = np.zeros(len(T)-1)
    #for k in range(N-1):
    #    timestep_[k] = Delay[k] - Delay[k+1]
    #timestep = np.abs(np.sum(timestep_)/(N-1))
    timestep = 2*0.001*10**(-3) / (299792458) #comment this out for data that has measured the delay
    print(timestep)
    FX = fft(X)[0:N//2 +1]
    FDelay = fftfreq(len(X),d=timestep)[0:N//2 + 1] #Just use the positiv value of the frequencies
    FDelay = np.abs(FDelay)
    #############################
    #For Data without measured delay, the stepsize has to be known though
    step_size = 0.001
    start = 0
    end = len(X)*step_size + start
    Delay = np.linspace(start, end, len(X))


    data = np.array([Delay, X])
    Fdata = np.array([FDelay,np.abs(FX)])
    data_name = ['X', 'FX', 'log(FX)']
    data_name_x = ['Delay / ps', 'Frequency / THz', 'log(Frequecy / THz']
    data_name_y = ['X', 'Fourier[X]', 'log(X)']
    fig, ax = plt.subplots(len(data_name),1, figsize=(16,8))
    if n == 0:
        print(FDelay)
    #print(Fdata)
    ######################
    #   Plotting
    ######################

    plot_data = [data, Fdata, Fdata]
    for i in range(len(data_name)):
        if data_name[i] == 'log(FX)':
            ax[i].plot(plot_data[i][0], plot_data[i][1], label=data_name[i])
            ax[i].set_yscale('log')
        else:
            ax[i].plot(plot_data[i][0], plot_data[i][1], label=data_name[i])
        ax[i].grid()
        ax[i].set_xlabel(data_name_x[i])
        ax[i].set_ylabel(data_name_y[i])
        ax[i].set_title(data_name[i])
        # ax[i].legend()

    plt.tight_layout()
    plt.savefig(savefiles[n])