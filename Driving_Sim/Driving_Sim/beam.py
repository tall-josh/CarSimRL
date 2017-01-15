# -*- coding: utf-8 -*-
import pygame
import constants as CONST
import math

class Beam(pygame.sprite.Sprite):
      # Sprite for the player
    def __init__(self, beam_idx):
        self.color = CONST.COLOR_BLUE
        self.beam_idx = beam_idx  
        self.x1 = 0
        self.y1 = 0
        self.dist = 0
            
    def update(self, anchorX, anchorY, anchor_deg, obstacle_list, lidar):      
        for obs in obstacle_list:
            self.color = CONST.COLOR_BLUE
            for step in lidar.increments:
                self.x1 = anchorX + (step * math.cos(math.radians(self.beam_idx*CONST.LIDAR_STEP - anchor_deg + lidar.start_ang_deg - 90)))
                self.y1 = anchorY + (step * math.sin(math.radians(self.beam_idx*CONST.LIDAR_STEP - anchor_deg + lidar.start_ang_deg - 90)))
                if obs.rect.collidepoint(self.x1, self.y1):
                    self.dist = step
                    obs.tag = True
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
                self.dist = step
        return 
        
                   
                    
#                    return (self.x1, self.y1, CONST.LIDAR_COLORS[CONST.URGENCY[self.urgency]], step)
#                    
#        return (self.x1, self.y1, CONST.LIDAR_COLORS[CONST.URGENCY[self.urgency]], self.range)
        