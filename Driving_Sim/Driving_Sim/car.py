# -*- coding: utf-8 -*-
import pygame
import random
import constants as CONST
import math
import game_art as art
import lidar

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
        self.command_array = (0,0,0) #(stearing, acceleation, breaking)
        self.sensor_data = []
        
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

#       Q-Learning Current Reward
        self.reward = 0
            
    def attachLidar(self, to_attach):
        self.lidar = to_attach
        
    def linearDistribution(self, x, in_min=0, in_max=255, out_min=-33, out_max=33):
        return (x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
        
    def inputCommand(self, steering_input):
        self.heading += self.linearDistribution(steering_input) * CONST.ONE_DEGREE
  
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

    def updateSensors(self, obstacles):
        self.sensor_data = self.lidar.update(self.rect.centerx, self.rect.centery, math.degrees(self.heading), obstacles)
    
    def inputComand(self, command_array):
        self.command_array = command_array
        
    def update(self):
       if all(x == 0 for x in self.command_array):
           self.doPID(1/CONST.SCREEN_FPS)
           self.speed = CONST.CAR_MAX_SPEED    
       else:
           #Reset command array
           self.command_array = (0,0,0)
    
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
           
        
    #        keystate = pygame.key.get_pressed()
#        if keystate:
#            #steering
#            if  keystate[pygame.K_LEFT]:
#                self.heading += CONST.CAR_ANGULAR_ACCEL
#            if keystate[pygame.K_RIGHT]:
#                self.heading -= CONST.CAR_ANGULAR_ACCEL
#            
#            #acceleration
#            if keystate[pygame.K_UP]:
#                self.speed += CONST.CAR_FORWARD_ACCEL
#                if self.speed > CONST.CAR_MAX_SPEED:
#                    self.speed = CONST.CAR_MAX_SPEED
#            if not keystate[pygame.K_UP]:
#                self.speed -= CONST.CAR_FORWARD_ACCEL
#                if self.speed < 0:
#                    self.speed = 0

            

        

       
            