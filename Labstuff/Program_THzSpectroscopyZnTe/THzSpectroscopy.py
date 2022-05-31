import os
import sys
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)
from Driver.controller_esp301 import *
from Driver.Program_LockIn import *
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import re

#%% Open Devices
SR830= SR830()
Stage = ESP301()
#%% Initialize

Stage.initialize()
Stage.wait_till_done()
#Stage.write('1VA10')
# Velocity of the stage: 1VAX with X mm/sec
time_con=7
SR830.setIT(i=time_con)
#Integration times are as follows:
#"10us:"0","30us":"1","100us":"2",
#"300us:"3","1ms":"4","3ms":"5",
#10ms:"6","30ms":"7","100ms":"8",
#"300ms:"9","1s":"10","3s":"11",
#10s:"12","30s":"13","100s":"14",
#300s:"15","1ks":"16","3ks":"17",
#10ks:"18","30ks":"19"}
SR830.setSens(i=11)
#Sensitivities are as follows:
#"2nV":"0","5nV":"1","10nV":"2","20nV":"3","50nV":"4","100nV":"5","200nV":"6","500nV":"7","1uV":"8",
#"2uV":"9","5uV":"10","10uV":"11","20uV":"12","50uV":"13","100uV":"14","200uV":"15","500uV":"16","1mV":"17",
#"2mV":"18","5mV":"19","10mV":"20","20mV":"21","50mV":"22","100mV":"23","200mV":"24","500mV":"25","1V":"26"}

#%% Live plotting function

plt.style.use('ggplot')

def live_plotter(x_vec,y1_data,line1,start,stop,identifier='',pause_time=0.001,Y_label='X (V)',X_label='stage position (mm)'):
    if line1==[]:
        # this is the call to matplotlib that allows dynamic plotting
        plt.ion()
        fig = plt.figure(figsize=(13,6))
        ax = fig.add_subplot(111)
        # create a variable for the line so we can later update it
        line1, = ax.plot(x_vec,y1_data,'-o',alpha=0.8)        
        #update plot label/title
        plt.xlabel(X_label)
        plt.ylabel(Y_label)
        #plt.title('Title: {}'.format(identifier))
        plt.show()
    
    # after the figure, axis, and line are created, we only need to update the y-data
    line1.set_ydata(y1_data)
    # adjust limits if new data goes beyond bounds
    #if np.min(y1_data)<=line1.axes.get_ylim()[0] or np.max(y1_data)>=line1.axes.get_ylim()[1]:
    #    plt.ylim([np.min(y1_data)-np.std(y1_data),np.max(y1_data)+np.std(y1_data)])
    plt.xlim(start-0.5*np.std(x_vec),stop+0.5*np.std(x_vec))

    # this pauses the data so the figure/axis can catch up - the amount of pause can be altered above
    plt.pause(pause_time)
    
    # return line so we can update it again in the next iteration
    return line1

#%% File name management

#Daytime for File name management. The Program will automatically create a folder for the current day if not present. The file will get a name according to
# the time it was taken
now = datetime.now()
current_day=now.strftime("%Y:%m:%d")
current_time = now.strftime("%H:%M:%S")
#time=[int(s) for s in current_time.split() if s.isdigit()]
Current_time=re.findall(r'\d+', current_time)
#day=[int(s) for s in current_day.split() if s.isdigit()]
day=re.findall(r'\d+', current_day)
#initial_dir='Desktop\test'

path = '/Users/Lab II/Desktop/Data_THz_Spectroscopy/'+str(day[0])+str(day[1])+str(day[2])
initial_dir=path+'/'+str(Current_time[0])+'_'+str(Current_time[1])+'_'+str(Current_time[2])
isExist = os.path.exists(path)
if not isExist:
    os.makedirs(path)
    
#Data Acquisition
plt.close('all')
start=0
stop=10
step=0.05 #step in millimeter
time_constant=[10e-6,30e-6,100e-6,300e-6,1e-3,3e-3,10e-3,30e-3,100e-3,300e-3,1,3,10,30,100,300,1000,3000,10000,30000]
stepcount=int((stop-start)/step)
x=np.array(np.linspace(start,stop,stepcount+1))
X=[]
Y=[]
R=[]
theta=[]
out=[]
line1=[]
pos=0
plot_y=np.empty(len(x))  # Plot_y is the array which will be live plotted. It is given by (out[pos])[i] and i determines the measured parameter
plot_y[:]=np.NaN
for i in x:
    if pos==0:
        f=open(initial_dir+'.txt','w')
        f.write('X \t Y \t R \t theta \n')
    #print(i)
    Stage.set_pos_abs(i)
    Stage.wait_till_done()
    time.sleep(float(5*time_constant[time_con]))  
    Measurement=SR830.get_All_Outputs()
    out.append(Measurement)   #all the lock in outputs will be connected in an array [X,Y,R,theta]
    plot_y[pos]=(out[pos])[0]
    #line1=live_plotter(x,np.array(X),line1)
    line1=live_plotter(x,plot_y,line1,start=start,stop=stop)
    plt.ylim(np.min(plot_y[0:pos+1])-np.std(plot_y[0:pos+1]),np.max(plot_y[0:pos+1])+np.std(plot_y[0:pos+1]))
    
    
    f.write('\n'+str(Measurement[0])+'\t'+str(Measurement[1])+'\t'+str(Measurement[2])+'\t'+str(Measurement[3]))
    #np.savetxt('test.txt', out, delimiter='\t',header='X\t Y\t R\t $\\theta$')
    pos+=1
f.close() # Be careful: If f.close() not excecuted, the measurent data is not updated and the information from the last measurement will be saved in the txt file!!!
    
 
#%% Close Devices
SR830.close()
Stage.close()
''





'''
B.initialize()
B.write('1VA1')
B.set_pos_abs(30)
B.wait_till_done()
B.close()
'''
"""
pos = B.get_pos()
new_pos = 30
B.set_pos_rel(new_pos)
B.wait_till_done()
print(pos)
B.close()
# esp = ESP301('/dev/ttyS3')
# return_message = esp.initialize()
# print(return_message)
# esp.close()
"""

