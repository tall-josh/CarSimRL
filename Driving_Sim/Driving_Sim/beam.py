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
        self.color = CONST.COLOR_RED
        for i in range(length//resolution):
            self.increments.append((i+1)*resolution)
            
        #self.x1 = self.x0 + (self.length * math.cos(math.radians(self.beam_idx*self.step_ang)))
        #self.y1 = self.y0 + (self.length * math.sin(math.radians(self.beam_idx*self.step_ang)))
        #self.image = pygame.draw.lines(self.image, CONST.COLOR_RED, False, [(self.x0,self.y0), (self.x1,self.y1)], 1)
 
    def setAnchorPoint(self, anchorX, anchorY, anchor_deg):
        self.x0 = anchorX
        self.y0 = anchorY
        self.deg = anchor_deg
        #self.x1 = self.x0 + (self.resolution * math.cos(math.radians(self.beam_idx*self.step_ang - self.deg + self.start_ang_deg - 90)))
        #self.y1 = self.y0 + (self.resolution * math.sin(math.radians(self.beam_idx*self.step_ang - self.deg + self.start_ang_deg - 90)))
        
    def update(self, obstacle_list):
        
        self. color = CONST.COLOR_RED
        for step in self.increments:
            
            #select color for different ranges
            if step > self.length * 0.25:
                self.color = CONST.COLOR_ORANGE
            if step > self.length * 0.5:
                self.color = CONST.COLOR_YELLOW
            if step > self.length * 0.75:
                self.color = CONST.COLOR_GREEN
                
            self.x1 = self.x0 + (step * math.cos(math.radians(self.beam_idx*self.step_ang - self.deg + self.start_ang_deg - 90)))
            self.y1 = self.y0 + (step * math.sin(math.radians(self.beam_idx*self.step_ang - self.deg + self.start_ang_deg - 90)))
            for obs in obstacle_list:
                if obs.rect.collidepoint(self.x1, self.y1):
                    return (self.x1, self.y1, self.color)
        
        return (self.x1, self.y1, self.color)
                
#        self.image = pygame.Surface((CONST.SCREEN_SIZE))
#        self.image.set_colorkey((0,0,0))
#        pygame.draw.lines(self.image, CONST.COLOR_RED, False, [(self.x0,self.y0), (self.x1,self.y1)], 1)
#        self.mask = pygame.mask.from_surface(self.image)
        