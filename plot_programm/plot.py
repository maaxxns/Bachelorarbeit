"""TO DO 
cut out the noise behind the signal
quadrate the fft result to get the intensity
integrate over the intensity and compare with other plots

"""




import numpy as np 
#import pandas as pd
import matplotlib.pyplot as plt
#from uncertainties.unumpy import uarray
#import uncertainties
#import matplotlib
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
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
    #Isulate the THz Pulse
    peak = find_peaks(X, distance=100000) #finds the highest peak and returns the index
    delay_peak = delay[peak[0]]
    start_pulse = np.where(delay >= delay_peak-5*10**(-12))[0][0]
    end_pulse = np.where(delay >= delay_peak + .7*10**(-12))[0][0]






    #############################
    #   FFT of all Data 

    N = len(X)
    timestep = np.mean(delay[11]-delay[10]) #timestep between two points. Just take 10th and 11th point because I dont want first and second
    FX = fft(X)[0:N//2 +1]
    FDelay = fftfreq(N,d=timestep)[0:N//2 + 1] #Just use the positiv value of the frequencies
    FDelay = np.abs(FDelay)

    index_normal = np.where(FDelay <= 5*10**12)
    #############################
    #   FFT of just the THz Pulse

    X_pulse = X[start_pulse:end_pulse]
    N_pulse = len(X_pulse)
    FX_pulse = fft(X_pulse)[0:N_pulse//2 +1]
    FDelay_pulse = fftfreq(N_pulse,d=timestep)[0:N_pulse//2 + 1] #Just use the positiv value of the frequencies
    FDelay_pulse = np.abs(FDelay_pulse)
    
    index_pulse = np.where(FDelay_pulse <= 5*10**12)
    
    #############################
    #   FFT of all Data with zeropadding

    FX_zeropadding = fft(np.concatenate((np.zeros(2000),X_pulse)))[0:len(np.concatenate((np.zeros(2000),X_pulse)))//2 +1]
    FDelay_zeropadding = fftfreq(len(np.concatenate((np.zeros(2000),X_pulse))),d=timestep)[0:len(np.concatenate((np.zeros(2000),X_pulse)))//2 + 1] #Just use the positiv value of the frequencies
    FDelay_zeropadding = np.abs(FDelay_zeropadding)

    index_zeropadding = np.where(FDelay_zeropadding <= 5*10**12) # index where to cut of the frequency in the plot. in this case its 5 THz



    #############################

    data_normal = np.array([delay*10**12, X]) #from seconds to pico seconds
    Fdata_normal = np.array([FDelay[index_normal]*10**(-12),np.abs(FX[index_normal])]) #from Hertz to Terahertz
    FData_zeropadding = np.array([FDelay_zeropadding[index_zeropadding] *10**(-12), np.abs(FX_zeropadding[index_zeropadding])])
    FData_pulse = np.array([FDelay_pulse[index_pulse]*10**(-12),np.abs(FX_pulse[index_pulse])])
    
    data_name_normal = ['X', 'FX', 'log(FX)']
    data_name_zeropadding = ['X', 'FX_zeropadding', 'log(FX_zeropadding)'] #if zeropadding isnt wanted just switch the names to none zero padding and the plot_data
    data_name_pulse = ['X', 'FX_pulse', 'log(FX_pulse)']
    
    data_name_x = ['Delay / ps', 'Frequency / THz', 'log(Frequecy / THz)']
    data_name_y = ['X(V)', 'Fourier[X]', 'log(Fourier[X])']


    ######################################
    #   make one big list with all data sets so we can iterate over it while plotting

    data_all = [[data_normal, Fdata_normal, Fdata_normal],[data_normal, FData_zeropadding, FData_zeropadding],[data_normal, FData_pulse, FData_pulse]]
    data_name_all = [data_name_normal, data_name_zeropadding, data_name_pulse]
    savename = ['normal', 'zeropadding', 'pulse']

    ######################
    #   Plotting
    ######################
    

    # maybe make plot that includes a x axis labeling with position aswell
    #First we make a plot with the data, the FFT of data and the log(FFT(data)) all in one file
    for k in range(len(data_all)):

        f, ax = plt.subplots(len(data_all[k]),1, figsize=(16,8), num=10)
        for i in range(len(data_name_all[k])):
            if data_name_all[k][i].find('log(') == 0:
                ax[i].plot(data_all[k][i][0], data_all[k][i][1], label=data_name_all[k][i])
                ax[i].set_yscale('log')


            else:
                ax[i].plot(data_all[k][i][0], data_all[k][i][1], label=data_name_all[k][i])


            if data_name_all[k][i] == data_name_pulse:
                ax[i].vlines(data_all[k][i][0][start_pulse], ymin=-0.00025, ymax= 0.00025,colors='r', label='cut off FFT')
                ax[i].vlines(data_all[k][i][0][end_pulse], ymin=-0.00025, ymax= 0.00025, colors='r')
                ax[i].legend()

            ax[i].grid()
            ax[i].set_xlabel(data_name_x[i])
            ax[i].set_ylabel(data_name_y[i])
            ax[i].set_title(data_name_all[k][i])


        plt.tight_layout()
        plt.savefig(filename[n].split('.')[0]+savename[k]+'.pdf')
        plt.close()

#######################################
        #      Then we make three seperate plots
#######################################


        for i in range(len(data_name_all[k])):
            fig = plt.figure(i)
            if data_name_all[k][i].find('log(') == 0:
                plt.plot(data_all[k][i][0], data_all[k][i][1], label=data_name_all[k][i])
                plt.yscale('log')
            else:
                plt.plot(data_all[k][i][0], data_all[k][i][1], label=data_name_all[k][i])
            
            
            plt.grid()
            plt.xlabel(data_name_x[i])
            plt.ylabel(data_name_y[i])
            plt.title(data_name_all[k][i])
            plt.tight_layout()
            plt.savefig(filename[n].split('.')[0] + savename[k]+ data_name_all[k][i]+'.pdf')
            plt.clf()
            plt.close(fig=i)