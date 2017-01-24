# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:22:46 2016

@author: Josh
"""

import pygame
import constants as CONST
import car
import obstacle
import game_art as art 
import lidar
import math
import numpy as np
import state_tracker as st
import random
import road

log_data = False
if log_data:
    file_states = open("states_file.txt", 'w')
    file_states.close()
    file_states = open("actions_file.txt", 'w')
    file_states.close()
    file_states = open("rewards_file.txt", 'w')
    file_states.close()


# Init pygame and create window
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode(CONST.SCREEN_SIZE)
screen.fill(CONST.COLOR_BLUE)
pygame.display.set_caption("My Game")
clock = pygame.time.Clock()

# Creating sprites
all_sprites = pygame.sprite.Group()

#create obstacles
obstacles = pygame.sprite.Group()


# Create Agent and attach sensor
car = car.Car(lane_idx=random.randint(0,2),  
              car_art = art.cars['player'])

lidar = lidar.Lidar(car.rect.centerx, car.rect.centery, math.degrees(car.heading))
car.attachLidar(lidar)
all_sprites.add(car)

# initalise 
state = st.StateTracker(CONST.STATE_MATRIX_SIZE[0], CONST.STATE_MATRIX_SIZE[1], CONST.STATE_MATRIX_SIZE[2])

# data logging
states = []
actions = []
rewards = []

def doObsMerge(merge_count):
    
    if ( (random.uniform(0,1) < CONST.MERGE_PROB) and
         (merge_count < CONST.MAX_NO_MERGES) ):
        print("Try to merge")
        
        rand_obs = random.randint(0,len(obstacles)-1)
        left_is_0_right_is_1 = random.radint(0,1)
        
        if left_is_0_right_is_1 == 0:
            if obstacles.sprites()[rand_obs].performMerge('left', obstacles):
            merge_count += 1
            print("merge left")
        
        else obstacles.sprites()[rand_obs].performMerge('right', obstacles):
            merge_count += 1
            print("merge right")
            

def initObstacles(dir_with, dir_against):
    for i in range(dir_with):
        obs = obstacle.Obstacle(random.randint(0,2),
                                "l_to_r",
                                art.cars['gray'],
                                car.rect.centerx, 
                                car.rect.centery, 
                                other_obs = obstacles)
        all_sprites.add(obs)
        obstacles.add(obs)
        
    for i in range(dir_against):
        obs = obstacle.Obstacle(random.randint(3,5),
                            "r_to_l",
                            art.cars['gray'],
                            car.rect.centerx, 
                            car.rect.centery, 
                            obstacles)

        all_sprites.add(obs)
        obstacles.add(obs)
    


def initSimulation(car, state):
    
#    global obstacels
#    global all_sprites
    obstacles.empty()
    all_sprites.empty()
    all_sprites.add(car)
    
    car.reInit(lane_idx = random.randint(0,2))
    
    initObstacles(CONST.OBS_L_TO_R,CONST.OBS_R_TO_L)    
        
    # initial state assumes the agent 
    # has been stationary for a short period
    car.updateSensors(obstacles)

    
#def setMerges(no_merge, travel_dir):
#    assert(travel_dir == 'l_to_r' or 'r_to_l')
#    if travel_dir == 'l_to_r':
#        
    
    
######### I'M GOING WHERE THE ACTION ISSSS!!!!! ################
bears_shit_in_woods = True
while bears_shit_in_woods:
    
    initSimulation(car, state)
    pigs_fly = False
    ticks = 0;
    
    merge_count = 0
    
    while not pigs_fly:
        #####  PYGAME HOUSE CLEANING  #####
        # keep loop time constant
        clock.tick(CONST.SCREEN_FPS)
        screen.fill(CONST.COLOR_BLACK)
     
        action_idx = 0
                ##### For it I decide to put some user input fuctionality  #####
        # Process input (events)
        for event in pygame.event.get():
            #check for closing window
            if event.type == pygame.QUIT:
                pigs_fly = True
                bears_shit_in_woods = False
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_UP:
                    print("Up")
                    action_idx = 1
                if event.key == pygame.K_DOWN:
                    print("Down")
                    action_idx = 2
                if event.key == pygame.K_LEFT:
                    print("Left")
                    action_idx = 3
                if event.key == pygame.K_RIGHT:
                    print("Right")
                    action_idx = 4
        
        ##### Take action #####
        #print("Action: {0}".format(CONST.ACTION_AND_COSTS[action_idx]))
        car.updateAction(action_idx)  # apply action selected above
        all_sprites.update()          # pygame updating sprites
        collisions = pygame.sprite.spritecollide(car, obstacles, False)    # Check for agent obstacle collisions
        ##### Observe new state (s') #####
        car.updateSensors(obstacles)   # sensor update      

        reward = car.reward 
        
         # Check for terminal states and override 
        # reward to teminal values if necessary
        if collisions:
            pigs_fly = True
            reward = CONST.REWARDS['terminal_crash']
            print("Terminal Crash")
        if car.at_goal:
            pigs_fly = True
            reward = CONST.REWARDS['terminal_goal']
            car.terminal = True
            print("Terminal Goal")
        
        states.append(car.sensor_data)
        rewards.append(reward)
        actions.append(action_idx)
        
        ##### MORE PYGAME HOUSE KEEPING #####
    #   kill obstacles if they are out of range
        for obs in obstacles:            
            if obs.out_of_range:
                obs.kill()
                
        doObsMerge(merge_count)
    
#       check if car is out of bounds
        if car.out_of_bounds:
            pigs_fly = True
            print("Out_Of_bounds")
                
    #    Draw / render
        all_sprites.draw(screen)
        #Drawing lane markers
        center_guard = CONST.LANES[3] + CONST.LANE_WIDTH//2
        for i in range(-4,3):
            thickness = 1
            color = CONST.COLOR_WHITE
            if i == -1: 
                thickness = 5
                color = CONST.COLOR_YELLOW
            pygame.draw.line(screen, color, (0, center_guard + (i*CONST.LANE_WIDTH)), (CONST.SCREEN_WIDTH, center_guard + (i*CONST.LANE_WIDTH)), thickness)
            
    #   DRAW MOST RECENT Lidar
        for beam in car.lidar.beams:
            pygame.draw.line(screen, beam.color, (beam.x1, beam.y1), (car.rect.centerx, car.rect.centery))
        
        # After everything, flip display
        pygame.display.flip()
        ticks += 1
        
    if log_data:
        # Data Logging    
        with open("states_file.txt", 'ab') as file:
            np.savetxt(file, states, delimiter=", ", header="start\n", footer="end\n", fmt='%.1i')
            file.close()
        with open("actions_file.txt", 'ab') as file:
            np.savetxt(file, actions, newline=", ", header="start\n", footer="end\n", fmt='%.1i')
            file.close()
        with open("rewards_file.txt", 'ab') as file:
            np.savetxt(file, rewards, newline=", ", header="start\n", footer="end\n", fmt='%.1i')
            file.close()

    if not bears_shit_in_woods: break
    
    

pygame.quit();
