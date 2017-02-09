# -*- coding: utf-8 -*-
"""
Created on Tue Feb  7 10:21:22 2017

@author: Josh
"""
import os
import numpy as np

def logData(fileNames, data):
    states0, actions, rewards, states1 = data[0], data[1], data[2], data[3]
        # Data Logging
        
    with open(fileNames["states0"], 'ab') as file:
        for frame in states0:
            np.savetxt(file, frame, delimiter=",", newline=os.linesep, header="", footer="", fmt='%.1f')
        file.close() 
    with open(fileNames["actions"], 'ab') as file:
        np.savetxt(file, actions, delimiter=',', newline=os.linesep, header="", footer="", fmt='%1f')
        file.close()
    with open(fileNames["rewards"], 'ab') as file:
#                file.write("{0}, ".format(rewards))
        np.savetxt(file, rewards, delimiter=',', newline=os.linesep, header="", footer="", fmt='%1f')
        file.close() 
    with open(fileNames["states1"], 'ab') as file:
        for frame in states1:
            np.savetxt(file, frame, delimiter=",", newline=os.linesep, header="", footer="", fmt='%.1f')
        file.close()    

def logDataInit(fileNames):
#    datetime_tag = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_states = open(fileNames["states0"], 'a')
    file_states.close()
    file_states = open(fileNames["actions"], 'a')
    file_states.close()
    file_states = open(fileNames["rewards"], 'a')
    file_states.close()
    file_states = open(fileNames["states1"], 'a')
    file_states.close()