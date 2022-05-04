import numpy as np 
import matplotlib.pyplot as plt
from uncertainties.unumpy import uarray
import uncertainties
import matplotlib


T, P, Zero, Delay, X, Y, R, theta, m = np.genfromtxt('002_EOS_FFT.txt', delimiter='\t', unpack=True) #unpack the txt file with pixel values
data = np.array([X,Y,R])
data_name = ['X', 'Y', 'R']
fig, ax = plt.subplots(3,1, figsize=(16,8))


for i in range(3):
    ax[i].plot(T, data[i], label=data_name[i])
    ax[i].grid()
    ax[i].set_xlabel('Time/s ')
    ax[i].set_ylabel('V?')
    #ax[i].legend()
    ax[i].set_title(data_name[i])
# 'ax[0].plot(T, X, label='X')
# ax[1].plot(T,Y, label='Y')
# ax[2].plot(T,R, label='R')
# fig.legend()
# fig.grid()
# plt.xlabel('Time/s ')
# plt.ylabel(r'$/degree$')'

plt.tight_layout()
plt.savefig('plot1.pdf')