import numpy as np 
#import pandas as pd
import matplotlib.pyplot as plt
#from uncertainties.unumpy import uarray
#import uncertainties
#import matplotlib
from scipy.fft import fft, fftfreq
#from tqdm import tqdm
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilenames
import os

root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
relative_path = 'plot/home/max/Documents/Bachelorarbeit/Bachelorarbeit/plot_data/'
filename = askopenfilenames(multiple=True)
filename = list(filename)
#print(filename) # show an "Open" dialog box and return the path to the selected file
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

for n in (range(len(filename))):
    t, p, delay, X, Y, R, theta = np.genfromtxt(filename[n], delimiter='\t', unpack=True, skip_header=1) #unpack the txt file with pixel values
                                                                                     #I dont know why it wants to unpack 9 values,but thats the reason for the variable m
    #############################
    #   FFT of Data 
    N  = len(X) # Number of Datapoints in X
    timestep = np.mean(delay[11]-delay[10]) #timestep between two points. Just take 10th and 11th point because I dont want first and second
    FX = fft(X)[0:N//2 +1]
    FDelay = fftfreq(N,d=timestep)[0:N//2 + 1] #Just use the positiv value of the frequencies
    FDelay = np.abs(FDelay)
    
    
    FX_zeropadding = fft(np.concatenate((np.zeros(2000),X)))[0:len(np.concatenate((np.zeros(2000),X)))//2 +1]
    FDelay_zeropadding = fftfreq(len(np.concatenate((np.zeros(2000),X))),d=timestep)[0:len(np.concatenate((np.zeros(2000),X)))//2 + 1] #Just use the positiv value of the frequencies
    FDelay_zeropading = np.abs(FDelay_zeropadding)

    #############################
    #For Data without measured delay, the stepsize has to be known though
    step_size = p[1]-p[0]
    start = p[0]
    end = p[N-1]
    #Delay = np.linspace(start, end, N)


    data = np.array([delay*10**12, X]) #from seconds to pico seconds
    Fdata = np.array([FDelay*10**(-12),np.abs(FX)]) #from Hertz to Terahertz
    FData_zeropadding = np.array([FDelay_zeropadding *10**(-12), np.abs(FX_zeropadding)])
    data_name = ['X', 'FX_zeropadding', 'log(FX_zeropadding)'] #if zeropadding isnt wanted just switch the names to none zero padding and the plot_data
    data_name_x = ['Delay / ps', 'Frequency / THz', 'log(Frequecy / THz)']
    data_name_y = ['X(V)', 'Fourier[X]', 'log(Fourier[X])']
    fig, ax = plt.subplots(len(data_name),1, figsize=(16,8))
    ######################
    #   Plotting
    ######################




    # maybe make plot that includes a x axis labeling with position aswell
    plot_data = [data, FData_zeropadding, FData_zeropadding]
    #plot_data = [data, Fdata, Fdata, FData_zeropadding, FData_zeropadding]
    #plot_data = [data, Fdata, Fdata]
    for i in range(len(data_name)):
        if data_name[i].find('log(') == 0:
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
'''    for i in range(len(data_name)):
        if data_name[i].find('log(') == 0:
            plt.plot(plot_data[i][0], plot_data[i][1], label=data_name[i])
            plt.yscale('log')
        else:
            plt.plot(plot_data[i][0], plot_data[i][1], label=data_name[i])
        plt.grid()
        plt.xlabel(data_name_x[i])
        plt.ylabel(data_name_y[i])
        plt.title(data_name[i])
        # ax[i].legend()
        plt.tight_layout()
        plt.savefig(savefiles[n])'''