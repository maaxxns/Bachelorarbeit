"""TO DO 

Vergleich mit paper

Inlcude new measurements with full power.
calculate the electric field

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

#just take the filename out of the whole path
for i in range(len(filename)):
    name = filename[i]
    name = name[name.find('daten'):]
    filename[i] = name.split('/')[1]

savefiles = list(np.zeros(len(filename)))

#make a directory for the plot.pdf and get the propper filenames
for i in range(len(filename)):
    savefiles[i] = filename[i]+'.pdf'
    path = 'daten/' +  filename[i].split('.')[0]
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)

peak_distances = []
FFT = []
timestep_ = []

#######################################
#   functions
#######################################

#   qudrate and 'integrate'


for n in (range(len(filename))):
    t, p, delay, X, Y, R, theta = np.genfromtxt( 'daten/' + filename[n], delimiter='\t', unpack=True, skip_header=1) #unpack the txt file with pixel values
                                                                                     #I dont know why it wants to unpack 9 values,but thats the reason for the variable m
    




    print('processing file: ' + filename[n] + '...')
    #############################
    #Isolate the THz Pulse
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
    
    data_name_x = ['Delay / ps', 'Frequency / THz', 'Frequecy / THz']
    data_name_y = ['X(V)', 'Fourier[X]', 'log(Fourier[X])']


    ######################################
    #   make one big list with all data sets so we can iterate over it while plotting

    data_all = [[data_normal, Fdata_normal, Fdata_normal],[data_normal, FData_zeropadding, FData_zeropadding],[data_normal, FData_pulse, FData_pulse]]
    data_name_all = [data_name_normal, data_name_zeropadding, data_name_pulse]
    savename = ['normal', 'zeropadding', 'pulse']
    

    #FFT.append(FData_zeropadding)

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


            if data_name_all[k] == data_name_pulse:
                if data_name_all[k][i] == data_name_pulse[0]:
                    ax[i].vlines(data_all[k][i][0][start_pulse], ymin=-0.00025, ymax= 0.00025,colors='r', label='cut off FFT')
                    ax[i].vlines(data_all[k][i][0][end_pulse], ymin=-0.00025, ymax= 0.00025, colors='r')
                    ax[i].legend()

            ax[i].grid()
            ax[i].set_xlabel(data_name_x[i])
            ax[i].set_ylabel(data_name_y[i])
            ax[i].set_title(data_name_all[k][i])


        plt.tight_layout()
        plt.savefig('daten/' + filename[n].split('.')[0] + '/' +filename[n].split('.')[0]+savename[k]+'.pdf')
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
            plt.savefig('daten/' + filename[n].split('.')[0] + '/' + filename[n].split('.')[0] + savename[k]+ data_name_all[k][i]+'.pdf')
            plt.clf()
            plt.close(fig=i)




    ###############################
    #   further calculations
    ###############################
        # - i need:
        # +quadrated fft value of every dataset
        # +distance between lowest and highest peak
        # +intergral of quadrated fft values

        ###############
        #first the distance between peak low and high
    timestep_.append(timestep)
    FFT.append(FData_zeropadding[1])
    peak_height =(np.abs(np.max(X) + np.min(X)))
    peak_distances.append(peak_height)

power_of_fft = []
for i in range(len(FFT)):
    power = FFT[i]**2
    power_of_fft.append(np.sum(power)*timestep_[i])
    
power_of_fft = np.array(power_of_fft)
pump_power = np.array([75, 58.2, 36.80, 26.5, 10.24, 7.28, 45.1, 259/2])

# Fluences was measured independetly on a seperate day than the real fluence measurments
# so maybe they are not that precise
Fluences = np.array([10.74, 8.38,4.53, 3.31, 1.47, 0.94 ,5.75,15.63])

Radius_Strahl_auf_Kristall = 0.343
area_beam_c  rytsal = np.pi*Radius_Strahl_auf_Kristall**2

pulse_energy = pump_power*2 / 1000
pulse_energy_per_area_measuredFluence = Fluences*2 / 1000 /area_beam_crytsal
pulse_energy_fluence = Fluences*2 / 1000 * area_beam_crytsal
conversion_effiency_power = power_of_fft/(pulse_energy *10**(-6))
conversion_effiency_fluence = power_of_fft/(pulse_energy_fluence *10**(-6))

pulse_energy_per_area_measuredPower = pulse_energy/area_beam_crytsal

###########################
#   peak distances plotted
###########################
print('Pulse energy used: ',pulse_energy)

fig , ((axis1, axis2_1),( axis3_1, axis4)) = plt.subplots(2, 2, figsize=(24,8))

if filename[0] == '11_46_58.txt' and filename[-1] == '16_12_43.txt':
    if len(pulse_energy) == len(peak_distances):
        axis1.plot(pulse_energy, np.array(peak_distances)/np.max(peak_distances),'rx' ,label='peak_distances')
    else:
        print('WARNING pump power array diffrent length then peak distances. subplimantary range(len(peakdistances)) was used')
        axis1.plot(range(len(peak_distances)), np.array(peak_distances)/np.max(peak_distances),'rx' ,label='peak_distances', )
    axis1.grid()
    axis1.legend()
    axis1.set_xlabel('pulse energy ' + r'$(\mu\mathrm{J})$')
    axis1.set_ylabel('percentage of maximum peak distance')
    axis1.set_title('Peak Distances with diffrent Pump Power')

    ##########################
    #   Power of electric fields plotted
    ##########################

    if len(pulse_energy_per_area_measuredFluence) == len(peak_distances):
        l1, = axis2_1.plot(pulse_energy_per_area_measuredFluence, power_of_fft,'k*')
        axis2_2 = axis2_1.twinx()                                                       #axis2_1 is power of electric field with measured fluence axis2_2 is its effiency
        l3, = axis2_2.plot(pulse_energy_per_area_measuredFluence, conversion_effiency_fluence, 'bo')
        axis2_1.set_xlabel('pulse energy per unit area ' + r'$(\mathrm{mJ}/\mathrm{cm}^2)$')

        l2, = axis3_1.plot(pulse_energy_per_area_measuredPower, power_of_fft, 'r*')
        axis3_2 = axis3_1.twinx()                                                      #axis3_1 is power of electric field with measured power axis3_2 is its effiency
        l4, = axis3_2.plot(pulse_energy_per_area_measuredPower, conversion_effiency_power, 'mx')
        axis3_1.set_xlabel('pulse energy per unit area ' + r'$(\mathrm{mJ}/\mathrm{cm}^2)$')
        axis3_2.set_ylabel('conversion effiency (arb. units)')
        
        
    else:
        print('WARNING pump power array diffre nt length then peak distances. subplimantary range(len(peakdistances)) was used')
        axis2.plot(range(len(peak_distances)), power_of_fft,'kx' ,label='power electric field')
        axis2.set_xlabel('range()')
    
    axis2_1.grid()
    axis2_1.legend([l1, l3], ['power electric field fluence measured','conversion effiency fluence measured'])
    axis2_1.set_title('With measured Fluence values')
    axis2_1.set_ylabel('Power(THz)/arb. units')
    axis2_2.set_ylabel('conversion effiency (arb. units)')

    axis3_1.grid()
    axis3_1.legend([l2,l4], ['power eletric field power measured', 'conversion effiency power measured'])
    axis3_1.set_title('With measured Power Values (Fluence calculated)')
    axis3_1.set_ylabel('Power(THz)/arb. units')

    l5, = axis4.plot(pulse_energy_per_area_measuredFluence, power_of_fft,'k*')
    axis4_2 = axis4.twinx()
    l6, = axis4_2.plot(pulse_energy_per_area_measuredFluence, conversion_effiency_fluence, 'bo')
    l7, = axis4.plot(pulse_energy_per_area_measuredPower, power_of_fft, 'r*')
    l8, = axis4_2.plot(pulse_energy_per_area_measuredPower, conversion_effiency_power, 'mx')

    axis4.set_xlabel('pulse energy per unit area ' + r'$(\mathrm{mJ}/\mathrm{cm}^2)$')
    axis4.set_ylabel('Power(THz)/arb. units')
    axis4_2.set_ylabel('conversion effiency (arb. units)')
    axis4.legend([l5,l6,l7,l8], ['power eletric field fluence measured', 'conversion effiency fluence measured', 'power eletric field power measured', 'conversion effiency power measured'])
    axis4.grid()
    
    plt.tight_layout()
    plt.savefig('daten/Fluence_comparission.pdf')
    plt.savefig('daten/Fluence_comparission.png')
    plt.close()

 


 ###########################