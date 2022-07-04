import numpy as np 
#import pandas as pd
import matplotlib.pyplot as plt
#from uncertainties.unumpy import uarray
from uncertainties import ufloat,unumpy
#import matplotlib
from scipy.fft import fft, fftfreq
from scipy.signal import find_peaks
#from tqdm import tqdm
from tkinter import Tk     # from tkinter import Tk for Python 3.x
from tkinter.filedialog import askopenfilenames
import os
from scipy.optimize import curve_fit



root = Tk()
root.withdraw() # we don't want a full GUI, so keep the root window from appearing
relative_path = 'plot/home/max/Documents/Bachelorarbeit/Bachelorarbeit/plot_data/'
filename = askopenfilenames(multiple=True)
filename = list(filename)

#just take the filename out of the whole path
for i in range(len(filename)):
    name = filename[i]
    name = name[name.find('daten'):]
    filename[i] = name.split('/')[1]

savefiles = list(np.zeros(len(filename)))

#make a directory for the plot.pdf and get the propper filenames
for i in range(len(filename)):
    savefiles[i] = filename[i]+'.pdf'
    path = 'daten/plots/' +  filename[i].split('.')[0]
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

peak_distances_1 = []
FFT_1 = []
timestep_1 = []

peak_distances_2 = []
FFT_2 = []
timestep_2 = []

#######################################
#   functions
#######################################

#calculate eletric field from paper:

def E(A_B, A,B):
    wavelength = 800 * 10**(-9)
    n_0 = 2.85
    r = 4.04 *10**(-12)
    L = 1 *10**(-3)
    eltricfield =  (A_B/A+B)*wavelength /(2* np.pi * n_0**3 * r * L) # in SI (V/m)
    eltricfield = eltricfield *10**(-5)# in kV/cm (1 V/m = 1/10 kV/cm)
    return eltricfield

def pulse_energy(power):
    return power/1000 * 2 #1000 pulses per second with double the energy because of the chopper

def linear(x, m, b):
    return m*x +b

X_A = np.genfromtxt( 'daten/eltric_field_data/' + 'A.txt', delimiter='\t', unpack=True, skip_header=1, usecols=0) #unpack the txt file with pixel values
A = ufloat(np.mean(X_A), np.std(X_A))
A = np.abs(A)
X_B = np.genfromtxt( 'daten/eltric_field_data/' + 'B.txt', delimiter='\t', unpack=True, skip_header=1, usecols=0) #unpack the txt file with pixel values
B = ufloat(np.mean(X_B), np.std(X_B))
B = np.abs(B)

pump_power_1 = np.array([75, 58.2, 36.80, 26.5, 10.24, 7.28, 45.1, 259/2])

pump_power_2 = np.array([135, 90.5, 81.6, 56.4, 24.6, 186.4])

pump_power = np.concatenate((pump_power_1, pump_power_2), axis=None)

fields = []
for n in (range(len(filename))):
    t, p, delay, X, Y, R, theta = np.genfromtxt( 'daten/' + filename[n], delimiter='\t', unpack=True, skip_header=1) #unpack the txt file with pixel values
                                                                                     #I dont know why it wants to unpack 9 values,but thats the reason for the variable m
    #############################
    #Isolate the THz Pulse
    peak = find_peaks(X, distance=100000) #finds the highest peak and returns the index
    A_B = X[peak[0]]
    fields.append(E(A_B, A, B))

fields = np.array(fields)
fields = fields + fields*0.15 # acccount for reflection losses as paper says

params, cov = curve_fit(linear, pump_power, unumpy.nominal_values(fields)[:,0])
x = np.linspace(np.min(pump_power), np.max(pump_power))

##########################################
#   plotting
##########################################
plt.errorbar(x = pump_power_1, y = unumpy.nominal_values(fields[:len(pump_power_1)][:,0]) , yerr=unumpy.std_devs(fields[:len(pump_power_1)][:,0]) ,fmt='rx',label='half pump power')
plt.errorbar(x = pump_power_2, y = unumpy.nominal_values(fields[len(pump_power_1):][:,0]) , yerr=unumpy.std_devs(fields[len(pump_power_1):][:,0]) ,fmt='bx',label='full pump power')
plt.plot(x, linear(x, *params), '-', label='linear Fit')
for i in range(len(fields)):
    print('pump power/mW: ', pump_power[i], ' field/(kV/cm): ', fields[i])

plt.grid()
plt.xlabel(r'$Pump \: power \, (\mathrm{mW})$')
plt.ylabel(r'$electric\: Field \,(\mathrm{kV}/\mathrm{cm})$')
plt.title('electric field per pump power')
plt.legend()
plt.savefig('daten/eltric_field_data/eltric_field.pdf')