# -*- coding: utf-8 -*-
import pygame
import random
import constants as CONST
import math

class Obstacle(pygame.sprite.Sprite):
    # Sprite for the player
    def __init__(self, img_file, playerX, playerY, lidar_range):
        pygame.sprite.Sprite.__init__(self)
        
        self.image = pygame.image.load(img_file)
        self.__image_master = self.image
        self.rect = self.image.get_rect()
        self.rect.center = (0,0)#(random.randint(-100, CONST.SCREEN_WIDTH + 100), random.randint(-100, CONST.SCREEN_HEIGHT+100))            
        self.heading = 0#random.randint(0,359)
        self.speed = 0#random.randint(CONST.OBSTACLE_MIN_SPEED, CONST.OBSTACLE_MAX_SPEED)
        #self.initState(playerX, playerY, lidar_range)
        self.initStateStatic()
        self.out_of_range = False
        self.tag = False #Flag to indicate if Obstacle is seen by lidar
     
    def initStateStatic(self):
        
        self.rect.center = (CONST.SCREEN_WIDTH//4, CONST.SCREEN_HEIGHT//2 + 200)
        self.heading = 0
        self.speed = 0
        
    def initState(self, playerX, playerY, lidar_range):
        
        self.rect.center = (random.randint(-100, CONST.SCREEN_WIDTH + 100), random.randint(-100, CONST.SCREEN_HEIGHT+100))            
        a = self.rect.centerx - playerX
        b = self.rect.centery - playerY
        dist_to_player = math.sqrt((a*a)+(b*b))
        if dist_to_player < lidar_range:
            self.initState(playerX, playerY, lidar_range)
        
        self.heading = random.randint(0,359)
        self.speed = random.randint(CONST.OBSTACLE_MIN_SPEED, CONST.OBSTACLE_MAX_SPEED)
    
        
    def update(self):
            
        self.velx = self.speed * (math.cos(self.heading))
        self.vely = self.speed * (math.sin(self.heading))
        
        #old centre to realign after rotation
        old_cen = self.rect.center
        #perform rotation
        self.image = pygame.transform.rotate(self.__image_master, math.degrees(self.heading))
        self.rect = self.image.get_rect()
        self.rect.center = old_cen
        
        self.rect.x += self.velx
        self.rect.y -= self.vely
        
        if (self.rect.left > CONST.SCREEN_WIDTH+100) or (self.rect.right < -100) or (self.rect.top > CONST.SCREEN_HEIGHT+100) or (self.rect.bottom < -100):
            self.out_of_range = True
        else:
            self.out_of_range = False
               
        
        

       
            