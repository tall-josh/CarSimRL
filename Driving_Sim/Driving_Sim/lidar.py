# -*- coding: utf-8 -*-
import pygame
import beam

class Lidar(pygame.sprite.Sprite):
    def __init__(self, length, number_of_beams, sweep_ang_deg, anchorX, anchorY, anchor_deg, resolution):
        self.beams = []
        self.closest_dist = length
        self.length = length
        for i in range(number_of_beams):
            b = beam.Beam(length, number_of_beams, sweep_ang_deg, i, anchorX, anchorY, anchor_deg, resolution)
            self.beams.append(b)
        
    def update(self, anchorX, anchorY, anchor_deg, obstacle_list):
        #updates beam array and also tracks the incoming data to keep track of the most 'urgent' obstacle
        self.closest_dist = self.length
        for b in self.beams:
            b.update(anchorX, anchorY, anchor_deg, obstacle_list)
            if b.dist < self.closest_dist:
                self.closest_dist = b.dist
    