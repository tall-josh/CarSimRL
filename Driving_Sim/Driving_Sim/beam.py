# -*- coding: utf-8 -*-
import pygame
import constants as CONST
import math

class Beam(pygame.sprite.Sprite):
      # Sprite for the player
    def __init__(self, max_range, number_of_beams, sweep_ang_deg, beam_idx, anchorX, anchorY, anchor_deg, resolution):
        self.start_ang_deg = 90 - (sweep_ang_deg/2)
        self.x0 = anchorX
        self.y0 = anchorY
        self.x1 = 0
        self.y1 = 0
        self.deg = anchor_deg
        self.max_range = max_range
        self.dist = max_range
        self.color = CONST.COLOR_BLUE
        self.resolution = resolution
        self.beam_idx = beam_idx
        self.step_ang = sweep_ang_deg / (number_of_beams-1)
        self.increments = []
        for i in range(max_range//resolution):
            self.increments.append((i+1)*resolution)
            
            
    def update(self, anchorX, anchorY, anchor_deg, obstacle_list):        
        for obs in obstacle_list:
            self.color = CONST.COLOR_BLUE
            for step in self.increments:
                self.x1 = anchorX + (step * math.cos(math.radians(self.beam_idx*self.step_ang - anchor_deg + self.start_ang_deg - 90)))
                self.y1 = anchorY + (step * math.sin(math.radians(self.beam_idx*self.step_ang - anchor_deg + self.start_ang_deg - 90)))
                if obs.rect.collidepoint(self.x1, self.y1):
                    self.dist = step
                    obs.tag = True
#                   select color for different ranges
                    if 0 < step <= self.max_range * 0.25:
                        self.color = CONST.COLOR_RED
                    if self.max_range * 0.25 < step <= self.max_range * 0.5:
                        self.color = CONST.COLOR_ORANGE
                    if self.max_range * 0.5 < step <= self.max_range * 0.75:
                        self.color = CONST.COLOR_YELLOW
                    if self.max_range * 0.75 < step <= self.max_range:
                        self.color = CONST.COLOR_GREEN
                    return
        self.dist = step
        return
        
                   
                    
#                    return (self.x1, self.y1, CONST.LIDAR_COLORS[CONST.URGENCY[self.urgency]], step)
#                    
#        return (self.x1, self.y1, CONST.LIDAR_COLORS[CONST.URGENCY[self.urgency]], self.range)
        