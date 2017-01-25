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
import numpy as np
import state_tracker as st
import copy
import deep_q_neural_network as dqnn
import tensorflow as tf

log_data = False
if log_data:
    file_states = open("states_file.txt", 'w')
    file_states.close()
    file_states = open("actions_file.txt", 'w')
    file_states.close()
    file_states = open("rewards_file.txt", 'w')
    file_states.close()

try: 
    tf.reset_default_graph()      
except:
    pass
#if dqnn.session._opened:
#    print("Closeing Existing tf.session")
#    dqnn.session.close()

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
state = st.StateTracker()

# data logging
states = []
actions = []
rewards = []

# super sloppy repeting code like this soz !soz.
def doObsMerge(merge_count):
    if ( (random.uniform(0,1) < CONST.MERGE_PROB) and
         (merge_count < CONST.MAX_NO_MERGES) ):
        
        rand_obs = random.randint(0,len(obstacles)-1)
        left_is_0_right_is_1 = random.randint(0,1)
        
        if left_is_0_right_is_1 == 0:
            if obstacles.sprites()[rand_obs].performMergeLeft(obstacles):
                merge_count += 1
        
        else:
            if obstacles.sprites()[rand_obs].performMergeRight(obstacles):
                merge_count += 1
            

def initObstacles(num_l_to_r, num_r_to_l):
    for i in range(num_l_to_r):
        obs = obstacle.Obstacle(random.randint(0,2),
                                "l_to_r",
                                art.cars['gray'],
                                car.rect.centerx, 
                                car.rect.centery, 
                                other_obs = obstacles)
        all_sprites.add(obs)
        obstacles.add(obs)
        
    for i in range(num_r_to_l):
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
    
    
    ########## I'M GOING WHERE THE ACTION ISSSS!!!!! ################
score = 0               # total score of the round
ticks  = 0              # number of iterations in each round will give up if > than...
epochs = 10000
gamma = 0.9
epsilon = 1
leave_program = False
batch_size = 20
buffer = 40
replay = []
h = 0
target_q = 0
reward = 0
qMax = 0
for i in range(epochs):
    
    initSimulation(car, state)
    pigs_fly = False
    ticks = 0;
    
    while not pigs_fly:
        #####  PYGAME HOUSE CLEANING  #####
        # keep loop time constant
        clock.tick(CONST.SCREEN_FPS)
        screen.fill(CONST.COLOR_BLACK)
    
        # Returns quality estimates for all posiable actions
        qMatrix = dqnn.getQMat(state.state.flatten())
        state_0_flat = state.state.flatten()
        
        ##### SELECT ACTION #####
        # select random action or use best action from qMatrix
        action_idx = 0
        epsilon_rand = False
##### UNCOMMENT WHEN YOU"RE DONE CHECKING CONTROLS OF CAR ######
        if (random.random() < epsilon):
            action_idx = random.randint(0,len(CONST.ACTION_AND_COSTS)-1)
            epsilon_rand = True
        else:
            action_idx = np.argmax(qMatrix)
       
        ##### Take action #####
        #print("Action: {0}".format(CONST.ACTION_AND_COSTS[action_idx]))
        car.updateAction(action_idx)  # apply action selected above
        all_sprites.update()          # pygame updating sprites
        collisions = pygame.sprite.spritecollide(car, obstacles, False)    # Check for agent obstacle collisions
        ##### Observe new state (s') #####
        car.updateSensors(obstacles)   # sensor update      
        state.update(car.sensor_data)  # update state with new data
    
        ##### GET maxQ' from DCNN #####   
        next_qMatrix = dqnn.getQMat(state.state.flatten())
        qMax = next_qMatrix[0][np.argmax(next_qMatrix)]
        
        ##### Generate target vector to train network #####
        target = copy.deepcopy(qMatrix)
        reward = car.reward 
         # Check for terminal states and override 
        # reward to teminal values if necessary
        if (collisions or (car.lane_idx < 0) or (car.lane_idx > 2)):
            pigs_fly = True
            reward = CONST.REWARDS['terminal_crash']
            print("Terminal Crash")
        if car.isAtGoal():
            pigs_fly = True
            reward = CONST.REWARDS['terminal_goal']
            car.terminal = True
            print("Terminal Goal")
        if len(replay) < buffer:
            replay.append((state_0_flat, action_idx, reward, state.state))
        else:
            if h < (buffer-1):
                h += 1
            else:
                h = 0
            # when replay buffer full, just overwrite from idx[0] rather than clearing and starting again
            state_1_flat = state.state.flatten()
            replay[h] = (state_0_flat, action_idx, reward, state_1_flat)
            
            batch = random.sample(replay, batch_size)
            target_batch = []
            for element in batch:
                old_state, action, reward, new_state = element
                q_mat_old = dqnn.getQMat(old_state.flatten())
                q_val_old = np.argmax(q_mat_old)
                
                q_mat_new = dqnn.getQMat(new_state.flatten())
                q_val_new = np.argmax(q_mat_new)
        
                target_q = reward + (gamma*qMax)
                target[0][action_idx] = target_q
                target_batch.append(target)
                
            dqnn.fitBatch([row[0] for row in batch], target_batch)
        print("Tick")
        if epsilon > 0.1:
            epsilon -= 1/epochs
        
        if ticks % 20 == 0:
            print("Ticks: ", i)
            print("target_q: {0} = reward: {1} + gamma:{2} * (qMax: {3})".format(target_q, reward, gamma, qMax))
            print("action_idx: {0}".format(action_idx))
            print("Q_mat: ", qMatrix)   
            
            
        ##### MORE PYGAME HOUSE KEEPING #####
    #   respawn obstacles if they are out of range
        for obs in obstacles:            
            if obs.out_of_range:
                obs.initState(car.rect.centerx, car.rect.centery, CONST.LIDAR_RANGE)
                print("Reload")
                
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
        
    #   Draw goal
        pygame.draw.circle(screen, CONST.COLOR_WHITE, (car.carrot), 10) 
        
    #   DRAW MOST RECENT Lidar
        for beam in car.lidar.beams:
            pygame.draw.line(screen, beam.color, (beam.x1, beam.y1), (car.rect.centerx, car.rect.centery))
    
        ##### For it I decide to put some user input fuctionality  #####
        # Process input (events)
        for event in pygame.event.get():
            #check for closing window
            if event.type == pygame.QUIT:
                pigs_fly = True
                leave_program = True
                dqnn.session.close()
            
        # After everything, flip display
        pygame.display.flip()
        ticks += 1
        #print("ticks: {0}".format(ticks))
        
    if leave_program: break

dqnn.session.close()
pygame.quit();
