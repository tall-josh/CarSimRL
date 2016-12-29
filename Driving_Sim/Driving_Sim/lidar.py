# -*- coding: utf-8 -*-
import pygame
import beam

class Lidar(pygame.sprite.Sprite):
    def __init__(self, length, number_of_beams, sweep_ang_deg, anchorX, anchorY, anchor_deg, resolution):
        self.beams = []
        for i in range(number_of_beams):
            b = beam.Beam(length, number_of_beams, sweep_ang_deg, i, anchorX, anchorY, anchor_deg, resolution)
            self.beams.append(b)
        
    def update(self, anchorX, anchorY, anchor_deg, obstacle_list):
        for b in self.beams:
            b.update(anchorX, anchorY, anchor_deg, obstacle_list)
    