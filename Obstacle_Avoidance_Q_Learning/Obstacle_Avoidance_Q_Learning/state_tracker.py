# -*- coding: utf-8 -*-

import constants as CONST
import numpy as np
import copy



class StateTracker():
    def __init__(self, depth, beams, increments):
        self.depth = depth
        self.beams = beams
        self.increments = increments
        #keeping track of oldest state so I can remove it 
        #before inserting the most recent state
        self.frame_history = np.zeros((depth, (beams*increments)))
        
        self.oldest_state_idx = len(self.frame_history) - 1
        self.state = np.zeros(( (beams * increments, )))
        #initalizing gray scale state matrix
        for scan in self.frame_history:
            self.state += scan
            
    # new_scan is a 2d numpy array representing the lidar one_hot array
    def update(self, new_scan):
        
        new_scan = new_scan.flatten()
        # sutract oldest scan fron state
        self.state -= self.frame_history[self.oldest_state_idx]

        # superimpose new_scan into state matrix
        self.state += new_scan
        
        # replace oldest scan with new_scan
        self.frame_history[self.oldest_state_idx] = copy.deepcopy(new_scan)
        
        # increment olderst_scan_idx
        self.oldest_state_idx += 1
        if self.oldest_state_idx >= len(self.frame_history):
            self.oldest_state_idx = 0
        
    def reset(self):
        array_dims = self.frame_history.shape
        self.frame_history = np.zeros(array_dims) 
        self.oldest_state_idx = len(self.frame_history) - 1
        self.state = np.zeros(array_dims[1])
        #initalizing gray scale state matrix
        for scan in self.frame_history:
            self.state += scan
            
    def details(self):
        print("depth: {0}, beams: {1}, increments: {2}".format(self.depth, self.beams, self.increments))
        print("oldest_idx: {0}".format(self.oldest_state_idx))
        print(self.frame_history[self.oldest_state_idx])
        