# -*- coding: utf-8 -*-
import random
import constants as CONST
import car
import math

class Obstacle(car.Car):        
    
    def isAwayFromPlayer(self, playerX, playerY, locX, locY):
        if abs(playerY - locY) < CONST.CAR_LENGTH: # in same column
            return abs(playerX - locX) > (2*CONST.CAR_LENGTH)
        else:
            return True
    
    def isAwayFromOtherObs(self, other_obs, locX, locY):
        for obs in other_obs:
            if (abs(obs.rect.center[0] - locX) < (1.1*CONST.CAR_LENGTH) and
                abs(obs.rect.center[1] - locY) < CONST.LANE_WIDTH*0.8):
                    return False                
        return True
        
    # Used to init a dynamic obstacle
    def reInitObs(self, playerX, playerY, other_obs):
        locX = 0
        locY = CONST.LANES[self.lane_idx]
        minX = -CONST.SCREEN_PADDING
        maxX = math.ceil(CONST.SCREEN_WIDTH * 0.75)
        self.heading = CONST.DIRECTIONS['l_to_r']
        if self.direction == 'r_to_l':
            self.heading = CONST.DIRECTIONS['r_to_l']
            self.goal_dist = self.goal_dist * -1
            minX = 50
            maxX = CONST.SCREEN_WIDTH + CONST.SCREEN_PADDING
        
        valid = False
        while not valid:
            locX = random.randint(minX, maxX)
            valid = (self.isAwayFromPlayer(playerX, playerY, locX, locY) and
                     self.isAwayFromOtherObs(other_obs, locX, locY))
                
        self.goal = ((locX + self.goal_dist), locY)
        self.rect.center = (locX, locY)
        self.speed = CONST.INIT_SPEED
        self.merge_left_possiable = False
        self.merge_right_possiable = False
    
    def checkMerge(self, other_obs):
        self.merge_left_possiable = True
        self.merge_right_possiable = True
        for obs in other_obs:
            if (obs.rect.center == self.rect.center or
                (self.lane_idx == 0 and self.direction == 'l_to_r') or
                (self.lane_idx == len(CONST.LANES) -1) and self.direction == 'r_to_l'):
                pass # we are looking at ourselves or we are in a sholder lane
            
            # Check to for obs my right AND that its not in an outside lane                
            elif (self.direction == 'l_to_r'):
                '''left is NOT ok'''
                if ((self.rect.center[1] - obs.rect.center[1] < (CONST.LANE_WIDTH*0.9)) and
                    abs(self.rect.center[0] - obs.rect.center[0] < CONST.CAR_SAFE_BUBBLE)):
                    self.merge_left_possiable = False
                    
                '''right is NOT ok'''
                if ((obs.rect.center[1] - self.rect.center[1] < (CONST.LANE_WIDTH*0.9)) and
                    abs(self.rect.center[0] - obs.rect.center[0] < CONST.CAR_SAFE_BUBBLE)):
                    self.merge_right_possiable = False
            else: ###  r_to_l
                '''left is NOT ok'''
                if ((obs.rect.center[1] - self.rect.center[1] < (CONST.LANE_WIDTH*0.9)) and
                    abs(self.rect.center[0] - obs.rect.center[0] < CONST.CAR_SAFE_BUBBLE)):
                    self.merge_left_possiable = False
                    
                '''right is NOT ok'''
                if ((self.rect.center[1] - obs.rect.center[1] < (CONST.LANE_WIDTH*0.9)) and
                    abs(self.rect.center[0] - obs.rect.center[0] < CONST.CAR_SAFE_BUBBLE)):
                    self.merge_right_possiable = False
        
        return (self.merge_left_possiable or self.merge_right_possiable)
    
    def performMergeLeft(self, obstacles):
        self.checkMerge(obstacles)
        if self.merge_left_possiable:
            self.updateAction(1)
            return True
            
    def performMergeRight(self, obstacles):
        self.checkMerge(obstacles)
        if self.merge_right_possiable:
            self.updateAction(2)
            return True
        else: return False
         


    
    # Sprite for the player
    def __init__(self, lane_idx, direction, car_art, playerX, playerY, other_obs=[]):
        super(Obstacle, self).__init__(lane_idx, car_art)
        self.out_of_range = False
        self.direction = direction
        self.tag = False #Flag to indicate if Obstacle is seen by lidar        
        self.reInitObs(playerX, playerY, other_obs)
               
        
        

       
            