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
    def reInit(self, xPos, lane_idx, initial_action = 0):
        
        self.rect.center = (xPos, CONST.LANES[lane_idx])
        self.carrot = (self.rect.x + 50, CONST.LANES[lane_idx])
        self.at_goal = False
        self.velx = 0
        self.vely = 0
        self.heading = 0
        self.lane_idx = lane_idx
        self.out_of_bounds = False
        self.sensor_data = np.zeros(CONST.LIDAR_DATA_SIZE) 
        #Q-Learning Current Reward

        self.reward = 0
        self.speed = CONST.INIT_SPEED + 1
        self.action_shift_regester = [initial_action for i in range(CONST.CAR_CONTROL_DAMPENING_DEPTH)]
        self.shift_reg_idx = 0
        self.current_action = initial_action


        
             
    # Sprite for the player
    def __init__(self, lane_idx, car_art):
        pygame.sprite.Sprite.__init__(self)
        self.lidar = None #need to attach using attachLidar(self, lidar)
        self.image = pygame.image.load(car_art)
        self.rect = self.image.get_rect() # get rect from pygame sprite object            
        self.__image_master = self.image
        self.speed = CONST.INIT_SPEED + 1
        self.carrot_dist = 100
        self.PID = (0.5, 0.1, 0)
        self.delta_heading = 0
        self.goal = self.rect.center[0] + 100
        self.goal_increment = 100
        self.goal_count = 1
        self.reInit(0, lane_idx)
        self.tail_gaiting = False
        self.action_shift_regester = []
        self.shift_reg_idx = 0
        self.current_action = 0
        
        
        
    def attachLidar(self, to_attach):
        self.lidar = to_attach
        
    def __updateShiftRegister(self, action):
        self.action_shift_regester[self.shift_reg_idx] = action
        self.shift_reg_idx = (self.shift_reg_idx + 1) % CONST.CAR_CONTROL_DAMPENING_DEPTH
        if all(actions == action for actions in self.action_shift_regester):
            self.current_action = action
        else:
            self.current_action = 0
            
        
    # Updates car's controles and allocates rewards asociated
    # with the action taken. (Note: This does NOT allocate
    # rewards asociated with colisions and goals)
    def updateAction(self, action, force=False):
        if force:
            self.current_action = action
        else:
            self.__updateShiftRegister(action)
            
        action_str = CONST.ACTION_AND_COSTS[self.current_action][0]
        
        if action_str == 'do_nothing':
            pass
        elif action_str == 'change_left':
            self.lane_idx -= 1
        elif action_str == 'change_right':
            self.lane_idx += 1
        elif action_str == 'break':
            self.speed -= CONST.CAR_FORWARD_ACCEL
        elif action_str == 'accelerate':
            self.speed += CONST.CAR_FORWARD_ACCEL
    
        # apply cost of action
        self.reward = CONST.ACTION_AND_COSTS[action][1]

    # Exhibits the go to goal behaviour         
    def doPID(self, delta_time):
        dx = self.carrot[0] - self.rect.centerx
        dy = self.rect.centery - self.carrot[1]
        rad_to_carrot = math.atan2(dy,dx)
        delta_rad =  rad_to_carrot - self.heading
        delta_rad = math.atan2(math.sin(delta_rad), math.cos(delta_rad))
        self.delta_heading += delta_rad
        self.heading += (self.PID[0] * delta_rad) + (self.PID[1] * delta_time * delta_time * self.delta_heading) + (self.PID[2] * delta_rad/delta_time)

    # Updates sensor data with and alocates 
    # reward s for obstale proximity
    def updateSensors(self, obstacles):
        self.lidar.update(self.rect.centerx, self.rect.centery, math.degrees(self.heading), obstacles)
        self.sensor_data = self.lidar.onehot
        self.tail_gaiting = self.lidar.tail_gating
    
    def isAtGoal(self):
        self.at_goal = (self.rect.x > CONST.SCREEN_WIDTH) 
#        self.at_goal = (self.rect.x > self.goal)
#        if self.at_goal: 
#            self.goal_count += 1
#            if self.goal_count % 20 == 0:
#                self.goal += self.goal_increment
#                
#        if self.goal > CONST.SCREEN_WIDTH:
#            self.goal = CONST.SCREEN_WIDTH
#            self.goal_increment = 0

        return self.at_goal
                
        
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
       
       self.carrot = (self.rect.x + self.carrot_dist, CONST.LANES[self.lane_idx])
       
       self.isOutOfBounds()
       #print("Heading", self.heading)
       
       
         
           

            

        

       
            