# -*- coding: utf-8 -*-
import pygame
import constants as CONST
import math
import game_art as art
import numpy as np

class Car(pygame.sprite.Sprite):
    
    ''' 
    Initializing variables, I have done this in a seperate
    method to the constructor so I can reInit the instance
    without having to create a whole new oject.
    '''
    def reInit(self, lane_idx):
        self.rect = self.image.get_rect() # get rect from pygame sprite object            
        self.rect.center = (0, CONST.LANES[lane_idx])
        self.goal = (self.rect.x + 50, CONST.LANES[lane_idx])
        self.at_goal = False
        self.velx = 0
        self.vely = 0
        self.heading = 0
        self.lane_idx = lane_idx
        self.out_of_bounds = False
        self.latest_action = 0          # Latest action updated in updateAction()
        self.sensor_data = np.zeros(CONST.LIDAR_DATA_SIZE) 
        #Q-Learning Current Reward
        self.reward = 0

        
             
    # Sprite for the player
    def __init__(self, lane_idx, car_art):
        pygame.sprite.Sprite.__init__(self)
        self.lidar = None #need to attach using attachLidar(self, lidar)
        self.image = pygame.image.load(car_art)
        self.__image_master = self.image
        self.speed = CONST.INIT_SPEED + 1
        self.goal_dist = 100
        self.PID = (0.5, 0.1, 0)
        self.delta_heading = 0
        self.reInit(lane_idx)
        
        
    def attachLidar(self, to_attach):
        self.lidar = to_attach
        
    # Updates car's controles and allocates rewards asociated
    # with the action taken. (Note: This does NOT allocate
    # rewards asociated with obstacel proximity)
    def updateAction(self, action):
        self.latest_action = action
        self.reward = CONST.ACTION_AND_COSTS[action][1]
        action_str = CONST.ACTION_AND_COSTS[action][0]

        if action_str == 'do_nothing':
            return            
        if action_str == 'change_left':
            self.lane_idx -= 1
        if action_str == 'change_right':
            self.lane_idx += 1
        if action_str == 'break':
            self.speed -= CONST.CAR_FORWARD_ACCEL
        if action_str == 'accelerate':
            self.speed += CONST.CAR_FORWARD_ACCEL

        #check lanes are valid
        if self.lane_idx < 0:
            self.reward = CONST.REWARDS['terminal_crash']
            self.lane_idx = 0
        if self.lane_idx > 2:
            self.reward = CONST.REWARDS['terminal_crash']
            self.lane_idx = 2
            
#        if action_str == 'do_nothing':
#            return            
#        if action_str == 'soft_left':
#            self.heading += 1*CONST.ONE_DEGREE            
#        if action_str == 'medium_left':
#            self.heading += 3*CONST.ONE_DEGREE            
#        if action_str == 'hard_left':
#            self.heading += 9*CONST.ONE_DEGREE            
#        if action_str == 'soft_right':
#            self.heading -= 1*CONST.ONE_DEGREE            
#        if action_str == 'medium_right':
#            self.heading -= 3*CONST.ONE_DEGREE            
#        if action_str == 'hard_right':
#            self.heading -= 9*CONST.ONE_DEGREE            
#        if action_str == 'soft_break':
#            self.speed -= 0.1            
#        if action_str == 'medium_break':
#            self.speed -= 0.3            
#        if action_str == 'hard_break':
#            self.speed -= 0.9            
#        if action_str == 'soft_acceleration':
#            self.speed += 0.1
#        if action_str == 'medium_acceleration':
#            self.speed += 0.3
#        if action_str == 'hard_acceleration':
#            self.speed += 0.9
            
    # Exhibits the go to goal behaviour         
    def doPID(self, delta_time):
        dx = self.goal[0] - self.rect.centerx
        dy = self.rect.centery - self.goal[1]
        rad_to_goal = math.atan2(dy,dx)
        delta_rad =  rad_to_goal - self.heading
        delta_rad = math.atan2(math.sin(delta_rad), math.cos(delta_rad))
        self.delta_heading += delta_rad
        self.heading += (self.PID[0] * delta_rad) + (self.PID[1] * delta_time * delta_time * self.delta_heading) + (self.PID[2] * delta_rad/delta_time)

    # Updates sensor data with and alocates 
    # reward s for obstale proximity
    def updateSensors(self, obstacles):
        self.lidar.update(self.rect.centerx, self.rect.centery, math.degrees(self.heading), obstacles)
        self.sensor_data = self.lidar.onehot
        self.reward += self.lidar.reward
    
    def isAtGoal(self):
        self.at_goal = (self.rect.x > CONST.SCREEN_WIDTH)
        
    def isOutOfBounds(self):
        if (self.rect.x < - CONST.SCREEN_PADDING or 
        self.rect.y < 0 or 
        self.rect.y > CONST.SCREEN_HEIGHT or
        self.rect.x > CONST.SCREEN_WIDTH + CONST.SCREEN_PADDING):
            self.out_of_bounds = True
    
    # applies control input and updates animation
    def update(self):
        
       self.doPID(1/CONST.SCREEN_FPS)
        
       if (self.speed > CONST.MAX_SPEED): self.speed = CONST.MAX_SPEED
       if (self.speed < 0): self.speed = 0
           
       self.velx = self.speed * (math.cos(self.heading))
       self.vely = self.speed * (math.sin(self.heading))
           
       self.heading = math.atan2(math.sin(self.heading), math.cos(self.heading))
                
       #old centre to realign after rotation
       old_cen = self.rect.center
       
       #perform rotation
       self.image = pygame.transform.rotate(self.__image_master, math.degrees(self.heading))
       self.rect = self.image.get_rect()
       self.rect.center = old_cen
        
       self.rect.x += self.velx
       self.rect.y -= self.vely
       
       self.goal = (self.rect.x + self.goal_dist, CONST.LANES[self.lane_idx])
       
       self.isAtGoal()
       self.isOutOfBounds()
       #print("Heading", self.heading)
       
           
         
           

            

        

       
            