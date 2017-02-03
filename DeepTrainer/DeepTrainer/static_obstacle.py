# -*- coding: utf-8 -*-
import pygame

class StaticObstacle(pygame.sprite.Sprite):
             
    # Sprite for the player
    def __init__(self, locX, locY, car_art):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load(car_art)
        self.rect = self.image.get_rect() # get rect from pygame sprite object            
        self.__image_master = self.image
        self.rect.center = (locX, locY)


    # applies control input and updates animation
    def update(self):
      pass