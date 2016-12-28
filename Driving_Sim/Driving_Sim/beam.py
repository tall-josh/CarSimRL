# -*- coding: utf-8 -*-
import pygame
import random
import constants as CONST
import math

class Beam(pygame.sprite.Sprite):
      # Sprite for the player
    def __init__(self, length, number_of_beams, sweep_ang_deg, beam_idx, anchorX, anchorY, anchor_deg, resolution):
        self.start_ang_deg = 90 - (sweep_ang_deg/2)
        self.x0 = anchorX
        self.y0 = anchorY
        self.x1 = 0
        self.y1 = 0
        self.deg = anchor_deg
        self.length = length
        self.resolution = resolution
        self.beam_idx = beam_idx
        self.step_ang = sweep_ang_deg / (number_of_beams-1)
        self.setAnchorPoint(anchorX, anchorY, anchor_deg)
        self.increments = []
        self.urgency = 0
        for i in range(length//resolution):
            self.increments.append((i+1)*resolution)
 
    def setAnchorPoint(self, anchorX, anchorY, anchor_deg):
        self.x0 = anchorX
        self.y0 = anchorY
        self.deg = anchor_deg
        
    def update(self, obstacle_list):
        
        for obs in obstacle_list:
            self.urgency = 0
            for step in self.increments:
                self.x1 = self.x0 + (step * math.cos(math.radians(self.beam_idx*self.step_ang - self.deg + self.start_ang_deg - 90)))
                self.y1 = self.y0 + (step * math.sin(math.radians(self.beam_idx*self.step_ang - self.deg + self.start_ang_deg - 90)))
                
                if obs.rect.collidepoint(self.x1, self.y1):
                    #select color for different ranges
                    if 0 < step <= self.length * 0.25:
                        self.urgency = 4
                    if self.length * 0.25 < step <= self.length * 0.5:
                        self.urgency = 3
                    if self.length * 0.5 < step <= self.length * 0.75:
                        self.urgency = 2
                    if self.length * 0.75 < step <= self.length:
                        self.urgency = 1
                    
                    obs.tag(self.urgency)
                    return (self.x1, self.y1, CONST.LIDAR_COLORS[CONST.URGENCY[self.urgency]])
                    
        obs.tag(self.urgency)
        return (self.x1, self.y1, CONST.LIDAR_COLORS[CONST.URGENCY[self.urgency]])
        