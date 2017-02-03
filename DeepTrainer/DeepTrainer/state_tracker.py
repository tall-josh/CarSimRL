# -*- coding: utf-8 -*-

import constants as CONST
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches



class StateTracker():
    def __init__(self):
        #keeping track of oldest state so I can remove it 
        #before inserting the most recent state
        self.frame_history = np.zeros(CONST.FRAME_HISTORY_SIZE)
        
        self.oldest_state_idx = 0
        self.idx_old_to_new = [i for i in range(len(self.frame_history))] #[0,1,2,..n]
        self.state = np.zeros(CONST.STATE_MATRIX_SIZE)
        #initalizing gray scale state matrix
        for scan in self.frame_history:
            self.state += scan
            
    def reset(self):
        self.frame_history = np.zeros(self.frame_history.shape) 
        self.oldest_state_idx = 0
        self.idx_old_to_new = [i for i in range(len(self.frame_history))] #[0,1,2,..n]
        self.state = np.zeros(self.state.shape)
        #initalizing gray scale state matrix
        for scan in self.frame_history:
            self.state += scan
            
    # new_scan is a 2d numpy array representing the lidar one_hot array
    def update(self, new_scan):
        
        #plt.imshow(new_scan, cmap=plt.cm.hot)
        #new_scan = new_scan.flatten()
        # sutract oldest scan fron state
#        self.state -= self.frame_history[self.oldest_state_idx]
        
        # superimpose new_scan into state matrix
#        self.state += new_scan
        
        # replace oldest scan with new_scan
        self.frame_history[self.oldest_state_idx] = new_scan
        
        self.state = np.zeros(self.state.shape)       
        weight_idx = 0
        for frame in self.frame_history:
            self.state += frame*CONST.HISTORY_WEIGHTS[weight_idx]
            weight_idx += 1
        # increment olderst_scan_idx
        self.oldest_state_idx = (self.oldest_state_idx - 1) % len(self.frame_history)
        for idx in self.idx_old_to_new:
            idx = (idx + 1) % len(self.idx_old_to_new)
    
    def __plotFrame(self, data):
        values = np.unique(data.ravel())
        im = plt.imshow(data, interpolation='none')
        colors = [im.cmap(im.norm(value)) for value in values]
        patches = [mpatches.Patch(color=colors[i], label="Level {l}".format(l=values[i])) for i in range(len(values))]
        plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0. )
        plt.show()
        
    
    def plotState(self, plt_state=True, plt_full_history=False):
        if plt_state:
            self.__plotFrame(self.state)
            
        if plt_full_history:
            count = len(self.idx_old_to_new)-1
            for idx in self.idx_old_to_new:
                print("Frame T-{0}: ".format(count))
                self.__plotFrame(self.frame_history[idx])
                count -= 1
            
        
        

