# -*- coding: utf-8 -*-
import pygame
import constants as CONST
import math
import numpy as np

class Beam(pygame.sprite.Sprite):
      # Sprite for the player
    def __init__(self, beam_idx):
        self.color = CONST.COLOR_BLUE
        self.beam_idx = beam_idx  
        self.x1 = 0
        self.y1 = 0
        self.dist = 0
        
    def update(self, anchorX, anchorY, anchor_deg, obstacle_list, sorted_idx_list, lidar): 
        
        for idx in sorted_idx_list:
            self.color = CONST.COLOR_BLUE
            
            #adding noise to signel
            noisy_increments = lidar.increments + np.random.normal(0,0.5, len(lidar.increments))
            for step in noisy_increments:
                self.x1 = anchorX + (step * math.cos(math.radians(self.beam_idx*CONST.LIDAR_STEP - anchor_deg + lidar.start_ang_deg - 90)))
                self.y1 = anchorY + (step * math.sin(math.radians(self.beam_idx*CONST.LIDAR_STEP - anchor_deg + lidar.start_ang_deg - 90)))
                self.dist = step
                if obstacle_list.sprites()[idx].rect.collidepoint(self.x1, self.y1):
                    self.dist = step
#                   select color for different ranges
                    if 0 < step <= CONST.LIDAR_RANGE * 0.25:
                        self.color = CONST.COLOR_RED
                    if CONST.LIDAR_RANGE * 0.25 < step <= CONST.LIDAR_RANGE * 0.5:
                        self.color = CONST.COLOR_ORANGE
                    if CONST.LIDAR_RANGE * 0.5 < step <= CONST.LIDAR_RANGE * 0.75:
                        self.color = CONST.COLOR_YELLOW
                    if CONST.LIDAR_RANGE * 0.75 < step < CONST.LIDAR_RANGE:
                        self.color = CONST.COLOR_GREEN
                    return
                elif self.y1 > CONST.LANES[len(CONST.LANES)-1] + CONST.LANE_WIDTH//2:
                    self.dist = step
                    return
                elif self.y1 < CONST.LANES[0] - (CONST.LANE_WIDTH//2):
                    self.dist = step    
                    return
            # if no object is struck by beam
            self.dist = CONST.LIDAR_RANGE + 1
        