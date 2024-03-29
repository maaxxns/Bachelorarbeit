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
#    print('A_B in func: ', A_B)
    wavelength = 800 * 10**(-9)
    if filename[0][0] == '1':
        n_0 = 2.85
        r = 4.04 *10**(-12) 
        L = 1 *10**(-3)
    if filename[0][0] == 'G':
        n_0 = 3.193
        r = 4.04 *10**(-12) 
        L = 0.3 *10**(-3)
        
    eltricfield =  (A_B/(2*(A+B)))*wavelength /(2* np.pi * n_0**3 * r * L) # in SI (V/m)
    eltricfield = eltricfield *10**(-5)# in kV/cm 
    return eltricfield

def I(E):
    c = 299792458
    e_0 = 8.8541878128*10**(-12)
    return 1/2 * c*e_0*E**2

def Power(I, r=None): #r is Radius of Spot in SI 
    if r==None:
        r=2.5*10**(-3)
        A = np.pi * r**2
    else:
        A = np.pi * r**2
    return I*A

def pulse_energy(power):
    return power/1000 * 2 #1000 pulses per second with double the energy because of the chopper

def linear(x, m, b):
    return m*x +b

def signaltonoise_dB(peak, noise,axis=0, ddof=0):
    noise = np.std(noise)
    return peak/noise

X_A = np.genfromtxt( 'daten/eltric_field_data/' + 'A.txt', delimiter='\t', unpack=True, skip_header=1, usecols=0) #unpack the txt file with  values
A = ufloat(np.mean(X_A), np.std(X_A))
A = np.abs(A)
X_B = np.genfromtxt( 'daten/eltric_field_data/' + 'B.txt', delimiter='\t', unpack=True, skip_header=1, usecols=0) #unpack the txt file with  values
B = ufloat(np.mean(X_B), np.std(X_B))
B = np.abs(B)

X_A_GaP = np.genfromtxt( 'daten/eltric_field_data/' + 'GaP_A.txt', delimiter='\t', unpack=True, skip_header=1, usecols=0)
A_GaP = ufloat(np.mean(X_A_GaP), np.std(X_A_GaP))
A_GaP = np.abs(A_GaP)
X_B_GaP = np.genfromtxt( 'daten/eltric_field_data/' + 'GaP_B.txt', delimiter='\t', unpack=True, skip_header=1, usecols=0)
B_GaP = ufloat(np.mean(X_B_GaP), np.std(X_B_GaP))
B_GaP = np.abs(B_GaP)
#print('B ', B_GaP, 'A ', A_GaP)
print(A_GaP, B_GaP) #A_Gap and B_GaP is from low power measurement
print(A, B) # A and B is from the full power measurment, which makes it way higher 
pump_power_1 = np.array([75, 58.2, 36.80, 26.5, 10.24, 7.28, 45.1, 259/2])

pump_power_2 = np.array([135, 90.5, 81.6, 56.4, 24.6, 186.4])

pump_power_GaP = np.array([74.4, 56.7, 46.0, 51.5, 35.6, 24.2, 124.2])

if filename[0][0] == '1':
    pump_power = np.concatenate((pump_power_1, pump_power_2), axis=None)
if filename[0][0] == 'G':
    pump_power = pump_power_GaP

fields = []
for n in (range(len(filename))):
    t, p, delay, X, Y, R, theta = np.genfromtxt( 'daten/' + filename[n], delimiter='\t', unpack=True, skip_header=1) #unpack the txt file with pixel values
                                                                                     #I dont know why it wants to unpack 9 values,but thats the reason for the variable m
    #############################
    #Isolate the THz Pulse
    peak = find_peaks(X, distance=100000) #finds the highest peak and returns the index
    A_B = X[peak[0]]
    print('A-B with peak func: ', A_B, 'A-B with max', np.max(X))
    print('file: ', filename[n], 'signal to noise', signaltonoise_dB(A_B, X[0:100]))
    if filename[n][0] == '1':
        fields.append(E(A_B, A_GaP, B_GaP))
    if filename[n][0] == '2':
        fields.append(E(A_B, A, B))
    if filename[0][0] == 'G':
        fields.append(E(A_B, A_GaP, B_GaP))
    
    


fields = np.array(fields)
#fields = fields + fields*0.15 # acccount for reflection losses as paper says

params, cov = curve_fit(linear, pump_power, unumpy.nominal_values(fields)[:,0])
x = np.linspace(np.min(pump_power), np.max(pump_power))

print('params: ', params,' +/- ',np.sqrt(np.diag(cov)), ' max field: ', np.max(fields), )
##########################################
#   plotting
##########################################
if filename[0][0] == '1':
    plt.errorbar(x = pump_power_1, y = unumpy.nominal_values(fields[:len(pump_power_1)][:,0]) , yerr=unumpy.std_devs(fields[:len(pump_power_1)][:,0]) ,color = 'k',ls='' ,marker='o',label='lower initial power')
    plt.errorbar(x = pump_power_2, y = unumpy.nominal_values(fields[len(pump_power_1):][:,0]) , yerr=unumpy.std_devs(fields[len(pump_power_1):][:,0]) ,color = ((132/255, 184/255, 25/255)),ls='',marker='*',label='full initial power')
    #plt.plot(x, linear(x, *params), '-', label='linear Fit')
if filename[0][0] == 'G':
    plt.errorbar(x = pump_power_GaP, y = unumpy.nominal_values(fields[:,0]) , yerr=unumpy.std_devs(fields[:,0]) ,color = 'k',ls='' ,marker='o',label='lower initial power')
    plt.plot(x, linear(x, *params), '-', label='linear fit')
for i in range(len(fields)):
    print('pump power (mW): ', pump_power[i], ' field (kV/cm): ', fields[i])
plt.xticks(size = 14)
plt.yticks(size = 14)
plt.grid()
plt.xlabel('pump power ' + r'$(\mathrm{mW})$', fontsize=14)
plt.ylabel('electric field '+ r'$(\mathrm{kV}/\mathrm{cm})$', fontsize=14)
#if filename[0][0] == '1':
##    plt.title('electric field ZnTe', fontsize = 24)
#if filename[0][0] == 'G':
#    plt.title('electric field GaP', fontsize = 24)
plt.legend(loc='upper left')
plt.tight_layout()
if filename[0][0] == '1':
    plt.savefig('daten/eltric_field_data/eltric_field_ZnTe.pdf')
if filename[0][0] == 'G':
    plt.savefig('daten/eltric_field_data/eltric_field_GaP.pdf')
plt.close()
##########################################
#   Power and Intensity
##########################################

def pulse_energy_to_power(E): 
    Radius_Strahl_auf_Kristall = 0.343
    area_beam_crytsal = np.pi*Radius_Strahl_auf_Kristall**2
    return E*area_beam_crytsal*500*np.exp(1)

def power_to_pulse_energy(P):
    Radius_Strahl_auf_Kristall = 0.343
    area_beam_crytsal = np.pi*Radius_Strahl_auf_Kristall**2
    return P/(500*area_beam_crytsal)* 1/np.exp(1)

intensity = I(fields) #Fields in kV/cm
power_THz = Power(intensity)
conversion_effiency = power_THz[:,0]/(pump_power*10**(-3)) # conversion effiency is weird
fields = fields*10**(-2)
for i in range(len(pump_power)):
    print('pump_power:', pump_power[i]*10**(-3), 'Power THz:', power_THz[i])
    print('pump power: ', pump_power[i], 'conversion: ', conversion_effiency[i])
print('maximum thz power: ', np.max(power_THz))
print('maximum field strength: ', np.max(fields)*10**2)
fig , (axis1) = plt.subplots(1, 1, figsize=(16,8),  constrained_layout=True)
for tick in axis1.xaxis.get_major_ticks():
    tick.label.set_fontsize(35) 
for tick in axis1.yaxis.get_major_ticks():
    tick.label.set_fontsize(35) 
axis2 = axis1.twinx()
conversion_effiency = conversion_effiency *10**6
axis2.errorbar(x = pump_power_2, y = unumpy.nominal_values(conversion_effiency[len(pump_power_1):]), yerr = unumpy.std_devs(conversion_effiency[len(pump_power_1):]),color='red',ls='',marker='x',label='c. e. full initial power')
axis2.errorbar(x = pump_power_1, y = unumpy.nominal_values(conversion_effiency[:len(pump_power_1)]), yerr = unumpy.std_devs(conversion_effiency[:len(pump_power_1)]),color='blue',ls='',marker='x',label='c. e. low initial power')
axis2.legend(loc = 'upper left', prop={'size': 18})
axis2.yaxis.set_tick_params(labelsize=35)
if filename[0][0] == '1':
    power_THz = power_THz*10**6 #convert to muW
    axis1.errorbar(x = pump_power_2, y = unumpy.nominal_values(power_THz[len(pump_power_1):][:,0]), yerr = unumpy.std_devs(power_THz[len(pump_power_1):][:,0]),color = ((132/255, 184/255, 25/255)),ls='',marker='*',label='full initial power')
    axis1.errorbar(x = pump_power_1, y = unumpy.nominal_values(power_THz[:len(pump_power_1)][:,0]), yerr = unumpy.std_devs(power_THz[:len(pump_power_1)][:,0]),color='k',ls='' ,marker='o', label='lower initial power') #10**(2) because of conversion in SI +3 for mW
    axis1.legend(loc=(0.009,0.73), prop={'size': 18})    
if filename[0][0] == 'G':
    power_THz = power_THz*10**6 #convert to muW
    axis1.errorbar(x = pump_power, y = unumpy.nominal_values(power_THz[:,0]), yerr = unumpy.std_devs(power_THz[:,0]),color = 'k',ls='',marker='o',label='lower initial power')
    axis1.legend(loc=(0.009,0.86), prop={'size': 18})
axis1.grid()
axis1.set_xlabel('pump power ' + r'$(\mathrm{mW})$', fontsize=35)
axis1.set_ylabel('power THz field ' + r'$(\mu\mathrm{W})$', fontsize=35)
#axis1.set_title('peak THz Power per pump power', fontsize=24)
axis2.set_ylabel('conversion efficiency ' + r'$ \cdot \,10^{-6}$', fontsize=35)


if filename[0][0] == '1':
    plt.savefig('daten/eltric_field_data/Powerznte.pdf')
if filename[0][0] == 'G':
    plt.savefig('daten/eltric_field_data/Powergap.pdf')
plt.close()

