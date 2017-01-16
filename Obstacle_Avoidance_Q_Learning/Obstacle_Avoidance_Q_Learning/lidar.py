# -*- coding: utf-8 -*-
import pygame
import beam
import constants as CONST
import numpy as np

class Lidar(pygame.sprite.Sprite):
    def __init__(self, anchorX, anchorY,anchor_deg):
        self.beams = []
        self.onehot = np.zeros(CONST.LIDAR_DATA_SIZE)
        self.closest_dist = CONST.LIDAR_RANGE
        
        self.start_ang_deg = 90 - (CONST.LIDAR_SWEEP/2)
        self.x0 = anchorX
        self.y0 = anchorY
        self.color = CONST.COLOR_BLUE
        self.increments = []
        self.reward = 0
        
        for i in range(CONST.LIDAR_COUNT):
            b = beam.Beam(i)
            self.beams.append(b)

        temp = 0
        for i in range((CONST.LIDAR_RANGE // CONST.LIDAR_RES)+1):
            self.increments.append(temp)
            temp += CONST.LIDAR_RES
        
    def getReward(self, closest_object):
        if 0 < closest_object <= CONST.LIDAR_RANGE * 0.25:
            return CONST.REWARDS["emergency"]
        if CONST.LIDAR_RANGE * 0.25 < closest_object <= CONST.LIDAR_RANGE * 0.5:
            return CONST.REWARDS["dangerous"]
        if CONST.LIDAR_RANGE * 0.5 < closest_object <= CONST.LIDAR_RANGE * 0.75:
            return CONST.REWARDS["uneasy"]
        if CONST.LIDAR_RANGE * 0.75 < closest_object < CONST.LIDAR_RANGE:
            return CONST.REWARDS["safe"]
        else:
            return CONST.REWARDS["out_of_range"]
            
    def update(self, anchorX, anchorY, anchor_deg, obstacle_list):
        #updates beam array and also tracks the incoming data to keep track of the most 'urgent' obstacle
        self.closest_dist = CONST.LIDAR_RANGE
        self.onehot = np.zeros(CONST.LIDAR_DATA_SIZE)
        
        for i in range(len(self.beams)):
            self.beams[i].update(anchorX, anchorY, anchor_deg, obstacle_list, self) 
            
            if self.beams[i].dist < CONST.LIDAR_RANGE:
                self.onehot[i][self.beams[i].dist // CONST.LIDAR_RES] = 1
            
            if self.beams[i].dist < self.closest_dist:                              
                self.closest_dist = self.beams[i].dist
                
        
        self.reward = self.getReward(self.closest_dist)
        self.onehot = self.onehot.flatten()
        