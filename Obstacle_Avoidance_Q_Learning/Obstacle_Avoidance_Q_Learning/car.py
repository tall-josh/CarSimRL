# -*- coding: utf-8 -*-
import pygame
import random
import constants as CONST
import math
import game_art as art
import lidar
import numpy as np

class Car(pygame.sprite.Sprite):
    # Sprite for the player
    def __init__(self):
        pygame.sprite.Sprite.__init__(self)
        self.lidar = None #need to attach using attachLidar(self, lidar)
        self.image = pygame.image.load(art.cars['player'])
        self.__image_master = self.image
        self.rect = self.image.get_rect()
        self.rect.center = (0,0)
        self.velx = 0
        self.vely = 0
        self.speed = 0
        self.heading = 0#math.radians(random.randint(0,359))
        self.goal = (0,0)
        self.at_goal = False
        self.out_of_bounds = False
        self.latest_action = 0
        self.sensor_data = np.zeros(CONST.LIDAR_DATA_SIZE)
        
        #Q-Learning Current Reward
        self.reward = 0
        
        goal_valid = False
        while not goal_valid:
            self.goal = (random.randint(10,CONST.SCREEN_WIDTH-10), random.randint(10,CONST.SCREEN_HEIGHT-10))
            self.rect.center = (random.randint(-50, CONST.SCREEN_WIDTH+50), random.randint(-50, CONST.SCREEN_HEIGHT+50))
            a = self.rect.centerx - self.goal[0]
            b = self.rect.centery - self.goal[1]
            # absolute worst case senario
            min_dist = math.sqrt((CONST.SCREEN_WIDTH/2)*(CONST.SCREEN_WIDTH/2) + (CONST.SCREEN_HEIGHT/2)*(CONST.SCREEN_HEIGHT/2))
            if (math.sqrt((a*a)+(b*b))) > min_dist:
                goal_valid = True
        
#       PID Setup
        self.P = 0.05
        self.I = 0
        self.D = 0
    
    def reInit(self, random):
        self.rect = self.image.get_rect()
        self.rect.center = (0,0)
        self.velx = 0
        self.vely = 0
        self.speed = 0
        self.heading = 0#math.radians(random.randint(0,359))
        self.goal = (0,0)
        self.at_goal = False
        self.out_of_bounds = False
        self.latest_action = 0
        self.sensor_data = np.zeros(CONST.LIDAR_DATA_SIZE) 
        
        #Q-Learning Current Reward
        self.reward = 0
        
        if random:
            goal_valid = False
            while not goal_valid:
                self.goal = (random.randint(10,CONST.SCREEN_WIDTH-10), random.randint(10,CONST.SCREEN_HEIGHT-10))
                self.rect.center = (random.randint(-50, CONST.SCREEN_WIDTH+50), random.randint(-50, CONST.SCREEN_HEIGHT+50))
                a = self.rect.centerx - self.goal[0]
                b = self.rect.centery - self.goal[1]
                # absolute worst case senario
                min_dist = math.sqrt((CONST.SCREEN_WIDTH/2)*(CONST.SCREEN_WIDTH/2) + (CONST.SCREEN_HEIGHT/2)*(CONST.SCREEN_HEIGHT/2))
                if (math.sqrt((a*a)+(b*b))) > min_dist:
                    goal_valid = True
        else:
             self.goal = (CONST.SCREEN_WIDTH//2 +100, CONST.SCREEN_HEIGHT//2)
             self.rect.center = (50, CONST.SCREEN_HEIGHT//2)
        
    def attachLidar(self, to_attach):
        self.lidar = to_attach
        
    def linearDistribution(self, x, in_min=0, in_max=255, out_min=-33, out_max=33):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
    # Updates car's controles and allocates rewards asociated
    # with the action taken. (Note: This does NOT allocate
    # rewards asociated with obstacel proximity)
    def updateAction(self, action):
        self.latest_action = action
        self.reward += CONST.ACTION_AND_COSTS[action][1]
        action_str = CONST.ACTION_AND_COSTS[action][0]
        if action_str == 'do_nothing':
            return            
        if action_str == 'soft_left':
            self.heading += 1*CONST.ONE_DEGREE            
        if action_str == 'medium_left':
            self.heading += 3*CONST.ONE_DEGREE            
        if action_str == 'hard_left':
            self.heading += 9*CONST.ONE_DEGREE            
        if action_str == 'soft_right':
            self.heading -= 1*CONST.ONE_DEGREE            
        if action_str == 'medium_right':
            self.heading -= 3*CONST.ONE_DEGREE            
        if action_str == 'hard_right':
            self.heading -= 9*CONST.ONE_DEGREE            
        if action_str == 'soft_break':
            self.speed -= 0.1            
        if action_str == 'medium_break':
            self.speed -= 0.3            
        if action_str == 'hard_break':
            self.speed -= 0.9            
        if action_str == 'soft_acceleration':
            self.speed += 0.1
        if action_str == 'medium_acceleration':
            self.speed += 0.3
        if action_str == 'hard_acceleration':
            self.speed += 0.9
            
  
    def doPID(self, delta_time):
        dx = self.goal[0] - self.rect.centerx
        dy = self.rect.centery - self.goal[1]
        rad_to_goal = math.atan2(dy,dx)
        delta_rad =  rad_to_goal - self.heading
        delta_rad = math.atan2(math.sin(delta_rad), math.cos(delta_rad))
        self.heading += (self.P * delta_rad) + (self.I * delta_time * delta_time) + (self.D * delta_rad/delta_time)
        #print('Head: {0}, ToGoal: {1}, Diff: {2}'.format(math.degrees(self.heading), math.degrees(rad_to_goal), (math.degrees(delta_rad))))
        if math.sqrt((dx*dx)+(dy*dy)) < 20:
            self.at_goal = True

    # Updates sensor data with and alocates 
    # rewards for obstale proximity
    def updateSensors(self, obstacles):
        self.lidar.update(self.rect.centerx, self.rect.centery, math.degrees(self.heading), obstacles)
        self.sensor_data = self.lidar.onehot
        self.reward += self.lidar.reward
    
    # applies control input and updates animation
    def update(self):
       
#      check if input has been made (negative reward to override controls)
       if self.latest_action == 0:
           self.doPID(1/CONST.SCREEN_FPS)
           self.speed += CONST.CAR_FORWARD_ACCEL
       
    
       if (self.speed > CONST.CAR_MAX_SPEED): self.speed = CONST.CAR_MAX_SPEED
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
       
       if (self.rect.x > CONST.SCREEN_WIDTH or 
       self.rect.x < 0 or 
       self.rect.y < 0 or 
       self.rect.y > CONST.SCREEN_HEIGHT):
           self.out_of_bounds = True
           

            

        

       
            