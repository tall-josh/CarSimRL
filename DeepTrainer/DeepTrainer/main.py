# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:22:46 2016

@author: Josh
"""

import pygame
import constants as CONST
import random
import car
import obstacle
import game_art as art 
import lidar
import math
import state_tracker as st
import numpy as np
import copy
import os
import static_obstacle as static_obs
import datetime as dt

log_data = False
load_data = False

assert not (log_data and load_data), "Cannot log and load at the same time."

if log_data:
    datetime_tag = dt.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    file_states = open("states0_{0}.txt".format(datetime_tag), 'a')
    file_states.close()
    file_states = open("actions_{0}.txt".format(datetime_tag), 'a')
    file_states.close()
    file_states = open("rewards_{0}.txt".format(datetime_tag), 'a')
    file_states.close()
    file_states = open("states1_{0}.txt".format(datetime_tag), 'a')
    file_states.close()

states0 = []
rewards = []
actions = []
states1 = []

if load_data:

    with open("states0_file.txt", 'rb') as file:
        states0 = np.genfromtxt(file, dtype=float, delimiter=',')
        sp = states0.shape
        states0 = np.reshape(states0, (sp[0]//50, 50, sp[1]))
        file.close()
    with open("states1_file.txt", 'rb') as file:
        states1 = np.genfromtxt(file, dtype=float, delimiter=',')
        sp = states1.shape
        states1 = np.reshape(states1, (sp[0]//50, 50, sp[1]))
        file.close()
    with open("rewards_file.txt", 'rb') as file:
        rewards = np.genfromtxt(file, dtype=float, delimiter=',')
        file.close()
    with open("actions_file.txt", 'rb') as file:
        actions = np.genfromtxt(file, dtype=float, delimiter=',')
        file.close()
        
        

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
car = car.Car(lane_idx=random.randint(CONST.CAR_LANE_MIN, CONST.CAR_LANE_MAX),
              car_art = art.cars['player'])

lidar = lidar.Lidar(car.rect.centerx, car.rect.centery, math.degrees(car.heading))
car.attachLidar(lidar)
all_sprites.add(car)

# initalise
state = st.StateTracker()

# data logging
states = []
actions = []
rewards = []

def doObsMerge(merge_count):
    if ( (random.uniform(0,1) < CONST.MERGE_PROB) and
         (merge_count < CONST.MAX_NO_MERGES) ):
        print("try")
        rand_obs = random.randint(0,len(obstacles)-1)
        left_is_0_right_is_1 = random.randint(0,1)

        if left_is_0_right_is_1 == 0:
            if obstacles.sprites()[rand_obs].performMergeLeft(obstacles):
                merge_count += 1

        else:
            if obstacles.sprites()[rand_obs].performMergeRight(obstacles):
                merge_count += 1

moving_obstacles = False # should not need to touch this. Is set in init(Static)Obstacels
def initStaticObstacles(xPos, lanes):
    global moving_obstacles
    moving_obstacles = False
    for i in range(len(xPos)):
        obs = static_obs.StaticObstacle(xPos[i], CONST.LANES[lanes[i]],
                                art.cars['gray'])
        all_sprites.add(obs)
        obstacles.add(obs)

def initObstacles(num_l_to_r, num_r_to_l):
    global moving_obstacles
    moving_obstacles = True
    for i in range(num_l_to_r):
        obs = obstacle.Obstacle(random.randint(CONST.OBS_LN_LtoR_MIN, CONST.OBS_LN_LtoR_MAX),
                                "l_to_r",
                                art.cars['gray'],
                                car.rect.centerx,
                                car.rect.centery,
                                other_obs = obstacles)
        all_sprites.addstate(obs)
        obstacles.add(obs)

    for i in range(num_r_to_l):
        obs = obstacle.Obstacle(random.randint(CONST.OBS_LN_RtoL_MIN,CONST.OBS_LN_RtoL_MAX),
                            "r_to_l",
                            art.cars['gray'],
                            car.rect.centerx,
                            car.rect.centery,
                            other_obs = obstacles)

        all_sprites.add(obs)
        obstacles.add(obs)



def initSimulation(car, state, random_start = False, random_obs = False):

    global obstacels
    global all_sprites
    obstacles.empty()
    all_sprites.empty()
    all_sprites.add(car)

    # This is a super hacky way to try and bias away from initialising the car in 'breaking' state
    # when filling replay buffer
    
#    random_start = random.randint(0,len(CONST.ACTION_NAMES)-1)
#    random_start = random.randint(0,len(CONST.ACTION_NAMES)-1) if random_start == 3 else random_start
    obs_x = [100,350,500]
    obs_lanes = [1,2,3]
    if random_obs:
        obs_x = np.random.uniform(low=200, high=CONST.SCREEN_WIDTH*0.8, size=3)
        obs_lanes = np.random.randint(low=1, high=4, size=3)
        
    rand_x = random.uniform(0, CONST.SCREEN_WIDTH*0.75)
    rand_y = random.randint(1,3)

    valid = False
    while not valid:
        valid = not ((rand_y == 1 and abs(rand_x - obs_x[1-1]) < 1.5*CONST.CAR_SAFE_BUBBLE) or
                     (rand_y == 2 and abs(rand_x - obs_x[2-1]) < 1.5*CONST.CAR_SAFE_BUBBLE) or
                     (rand_y == 3 and abs(rand_x - obs_x[3-1]) < 1.5*CONST.CAR_SAFE_BUBBLE))
        rand_x = random.uniform(0, CONST.SCREEN_WIDTH*0.75)
        rand_y = random.randint(1,3)
        
    if random_start:
        car.reInit(rand_x, rand_y)
    else:
        car.reInit(-50, 2)

    #initObstacles(CONST.OBS_L_TO_R,CONST.OBS_R_TO_L)
    initStaticObstacles(xPos=obs_x, lanes=obs_lanes)

# Initial state assumes the agent
# Has been stationary for a short period
    car.updateSensors(obstacles)
    
    ########## I'M GOING WHERE THE ACTION ISSSS!!!!! ################
run_count = 0               # total score of the round

bears_shit_in_woods = True
batch_size = 50
buffer = 100
merge_count = 0

while bears_shit_in_woods:
    
    initSimulation(car, state, random_obs=True)
    pigs_fly = False
    ticks = 0
    run_count += 1
    print("run: ", run_count)
    
    while not pigs_fly:
        #####  PYGAME HOUSE KEEPING  #####
        # keep loop time constant
        clock.tick(CONST.SCREEN_FPS)
        screen.fill(CONST.COLOR_BLACK)

# Save state0, action
        states0.append(copy.deepcopy(state.state))
        
        ##### SELECT ACTION #####
        action_idx = 0
        # Process input (events)
        for event in pygame.event.get():
            #check for closing window
            if event.type == pygame.QUIT:
                bears_shit_in_woods = False     # leave program
                pigs_fly = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    action_idx = 1
                if event.key == pygame.K_DOWN:
                    action_idx = 2
                if event.key == pygame.K_LEFT:
                    action_idx = 3
                if event.key == pygame.K_RIGHT:
                    action_idx = 4
        

        
        car.updateAction(action_idx)  # apply action selected above
        all_sprites.update()          # pygame updating sprites
        collisions = pygame.sprite.spritecollide(car, obstacles, False)    # Check for agent obstacle collisions
        ##### Observe new state (s') #####
        car.updateSensors(obstacles)   # sensor update     
        state.update(car.sensor_data)  # update state with new data
        
# Save state1, action
        states1.append(copy.deepcopy(state.state))
        actions.append(copy.deepcopy(action_idx))

        reward = car.reward 
        
        # Check for additional penalties from dangerous driving        
#        if car.tail_gaiting:
#            reward = CONST.REWARDS['tail_gate']
#            print('tail_gate')
#        
#        if car.lane_idx == 0:
#            reward = CONST.REWARDS['on_sholder']
#            print('on_sholder')
         # Check for terminal states and override 
        # reward to teminal values if necessary
        if (collisions or car.out_of_bounds):
            pigs_fly = True
            reward = CONST.REWARDS['terminal_crash']
            print('terminal_crash')
        
        if car.isAtGoal():
            pigs_fly = True
            reward = CONST.REWARDS['terminal_goal']
            car.terminal = True
            print('terminal_goal')
         
# Save reward
        rewards.append(copy.deepcopy(reward))
#        doObsMerge(merge_count)
        
#       Draw / render
        all_sprites.draw(screen)
        #Drawing lane markers
        center_guard = CONST.LANES[3] + CONST.LANE_WIDTH//2
        color = CONST.COLOR_ORANGE
        for lane in CONST.LANES:
            pygame.draw.line(screen, color, (0, lane-CONST.LANE_WIDTH//2), (CONST.SCREEN_WIDTH,  lane-CONST.LANE_WIDTH//2))
            color = CONST.COLOR_WHITE
        pygame.draw.line(screen, CONST.COLOR_ORANGE, (0, CONST.LANES[len(CONST.LANES)-1] + CONST.LANE_WIDTH//2), (CONST.SCREEN_WIDTH,  CONST.LANES[len(CONST.LANES)-1] + CONST.LANE_WIDTH//2))
                
#       Draw carrot
        pygame.draw.circle(screen, CONST.COLOR_ORANGE, (car.carrot), 5) 
        
#       DRAW MOST RECENT Lidar
        for beam in car.lidar.beams:
            pygame.draw.line(screen, beam.color, (beam.x1, beam.y1), (car.rect.centerx, car.rect.centery))
            
        # After everything, flip display
        pygame.display.flip()
################################  
    
    if log_data:
        # Data Logging    
            with open("states0_file.txt", 'ab') as file:
                for frame in states0:
                    np.savetxt(file, frame, delimiter=",", newline=os.linesep, header="", footer="", fmt='%.1f')
                file.close() 
            with open("actions_file.txt", 'ab') as file:
                np.savetxt(file, actions, delimiter=',', newline=os.linesep, header="", footer="", fmt='%1f')
                file.close()
            with open("rewards_file.txt", 'ab') as file:
#                file.write("{0}, ".format(rewards))
                np.savetxt(file, rewards, delimiter=',', newline=os.linesep, header="", footer="", fmt='%1f')
                file.close() 
            with open("states1_file.txt", 'ab') as file:
                for frame in states1:
                    np.savetxt(file, frame, delimiter=",", newline=os.linesep, header="", footer="", fmt='%.1f')
                file.close()    
    
    states0.clear()
    rewards.clear()
    actions.clear()
    states1.clear()

    if not bears_shit_in_woods: break

pygame.quit();
