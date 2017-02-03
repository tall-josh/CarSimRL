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
        self.tail_gating = False
        
        for i in range(CONST.LIDAR_COUNT):
            b = beam.Beam(i)
            self.beams.append(b)

        # increments are used for colsions detection 
        # between beams and obstacles.
        temp = 0
        for i in range((CONST.LIDAR_RANGE // CONST.LIDAR_RES)+1):
            self.increments.append(temp)
            temp += CONST.LIDAR_RES
            
    # uses the closest object detected by the lidar to determine 
    # the reward (penality)
#    def getReward(self, closest_object):
#        if 0 < closest_object <= CONST.LIDAR_RANGE * 0.25:
#            return CONST.REWARDS["emergency"]
#        if CONST.LIDAR_RANGE * 0.25 < closest_object <= CONST.LIDAR_RANGE * 0.5:
#            return CONST.REWARDS["dangerous"]
#        if CONST.LIDAR_RANGE * 0.5 < closest_object <= CONST.LIDAR_RANGE * 0.75:
#            return CONST.REWARDS["uneasy"]
#        if CONST.LIDAR_RANGE * 0.75 < closest_object < CONST.LIDAR_RANGE:
#            return CONST.REWARDS["safe"]
#        else:
#            return CONST.REWARDS["out_of_range"]

    def sortNearestObstacles(self, cenX, cenY, obstacle_list):
        sqr_dists = []
        idx = 0
        for obs in obstacle_list:
            dist = (cenX-obs.rect.center[0])**2 + (cenY-obs.rect.center[1])**2
            sqr_dists.append((dist, idx))
            idx += 1
        sqr_dists.sort()
        idxs = [i[1] for i in sqr_dists]
        return idxs
        #obstacle_list.sprites() = [obstacle_list.sprites()[i] for i in idxs]

    def update(self, anchorX, anchorY, anchor_deg, obstacle_list):
        self.tail_gating = False
        #updates beam array and also tracks the incoming data to keep track of the most 'urgent' obstacle
        self.closest_dist = CONST.LIDAR_RANGE
        self.onehot = np.zeros(CONST.LIDAR_DATA_SIZE)
        
        #sote indexes of obstacles from nearest to furthest
        sorted_idx_list = self.sortNearestObstacles(anchorX, anchorY, obstacle_list)
        
        
        for i in range(len(self.beams)):
            self.beams[i].update(anchorX, anchorY, anchor_deg, obstacle_list, sorted_idx_list, self) 
            
            if self.beams[i].dist < CONST.LIDAR_RANGE:
                self.onehot[i][int(self.beams[i].dist / CONST.LIDAR_RES) -1] = 1
            
            if self.beams[i].dist < self.closest_dist:                              
                self.closest_dist = self.beams[i].dist
            if ((i > (len(self.beams)//2) - 2) and
                (i < (len(self.beams)//2) + 2) and
                (self.beams[i].dist < CONST.TAIL_GATE_DIST)):
                self.tail_gating = True
                
        self.onehot = self.onehot
        