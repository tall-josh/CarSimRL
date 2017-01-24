# -*- coding: utf-8 -*-
"""
Created on Sat Jan 21 22:32:57 2017

@author: Josh
"""

import numpy as np

states = np.array([[1,2,3],[4,5,6]])
file_states = open("test.txt", 'w')
file_states.close()

with open("test.txt", 'ab') as file_states:
    np.savetxt(file_states, states, fmt='%i', newline=", ", delimiter="-", header = "Start", footer="End\n")

file_states.close()
