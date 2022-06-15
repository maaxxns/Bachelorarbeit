import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from uncertainties.unumpy import uarray
import uncertainties
import matplotlib
from scipy.fft import fft, fftfreq
from tqdm import tqdm
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilenames
import os

root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
relative_path = 'plot/home/max/Documents/Bachelorarbeit/Bachelorarbeit/plot_data/'
filename = askopenfilenames(multiple=True)
filename = list(filename)
print(filename) # show an "Open" dialog box and return the path to the selected file
for i in range(len(filename)):
    name = filename[i]
    name = name[name.find('daten'):]
    filename[i] = name    
print(filename)

#files = ['13_19_29.txt', '13_21_32.txt', '13_23_05.txt', '13_24_16.txt', '14_07_55.txt', '14_13_51.txt', '15_17_04.txt', '16_01_01.txt']
savefiles = list(np.zeros(len(filename)))

for i in range(len(filename)):
    savefiles[i] = filename[i]+'.pdf'

#savefiles = ['plot14_45_16.pdf', 'plot16_27_14.pdf', 'plot16_21_02.pdf', 'plot16_18_12.pdf']

for n in tqdm(range(len(filename))):
    t, p, delay, X, Y, R, theta = np.genfromtxt(filename[n], delimiter='\t', unpack=True, skip_header=1) #unpack the txt file with pixel values
                                                                                     #I dont know why it wants to unpack 9 values,but thats the reason for the variable m
    #############################
    #   FFT of Data 
    N  = len(X) # Number of Datapoints in X
    #timestep_ = np.zeros(len(T)-1)
    #for k in range(N-1):
    #    timestep_[k] = Delay[k] - Delay[k+1]
    #timestep = np.abs(np.sum(timestep_)/(N-1))
    #timestep = 2*0.001*10**(-3) / (299792458) #comment this out for data that has measured the delay
    timestep = np.mean(delay)
    FX = fft(X)[0:N//2 +1]
    FDelay = fftfreq(N,d=timestep)[0:N//2 + 1] #Just use the positiv value of the frequencies
    FDelay = np.abs(FDelay)
    #############################
    #For Data without measured delay, the stepsize has to be known though
    step_size = p[1]-p[0]
    start = p[0]
    end = p[N-1]
    #Delay = np.linspace(start, end, N)


    data = np.array([delay, X])
    Fdata = np.array([FDelay,np.abs(FX)])
    data_name = ['X', 'FX', 'log(FX)']
    data_name_x = ['Delay / ps', 'Frequency / THz', 'log(Frequecy / THz']
    data_name_y = ['X', 'Fourier[X]', 'log(X)']
    fig, ax = plt.subplots(len(data_name),1, figsize=(16,8))
    ######################
    #   Plotting
    ######################




    # maybe make plot that includes a x axis labeling with position aswell
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