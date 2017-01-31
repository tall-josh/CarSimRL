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
import static_obstacle as static_obs

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

        

def initSimulation(car, state, filling_buffer = False):
    
    global obstacels
    global all_sprites
    obstacles.empty()
    all_sprites.empty()
    all_sprites.add(car)
    
    # This is a super hacky way to try and bias away from initialising the car in 'breaking' state
    # when filling replay buffer
#    random_start = random.randint(0,len(CONST.ACTION_NAMES)-1)
#    random_start = random.randint(0,len(CONST.ACTION_NAMES)-1) if random_start == 3 else random_start
    rand_x = random.uniform(0, CONST.SCREEN_WIDTH*0.75)
    rand_y = random.randint(1,3)
    
    if filling_buffer:
        car.reInit(rand_x, rand_y)
    else:
        car.reInit(0, 2)
    
    #initObstacles(CONST.OBS_L_TO_R,CONST.OBS_R_TO_L)    
    initStaticObstacles(xPos=[100,350,500], lanes=[1,2,3])
        
    # initial state assumes the agent 
    # has been stationary for a short period
    car.updateSensors(obstacles)
    
    
    ########## I'M GOING WHERE THE ACTION ISSSS!!!!! ################

score = 0               # total score of the round
ticks  = 0              # number of iterations in each round will give up if > than...
epochs = 50000
gamma = 0.9
epsilon = 1
leave_program = False
batch_size = 15
buffer = 10000
replay = []
h = 0
target_q = 0
reward = 0
qMax = 0
merge_count = 0
epoch_cnt = 0
episode = 0
for i in range(epochs):
    
    initSimulation(car, state, filling_buffer = True if len(replay) < buffer else False)
    pigs_fly = False
    ticks = 0
    
    while not pigs_fly:
        episode += 1 # for printing to screen
        ticks += 1  
        #####  PYGAME HOUSE KEEPING  #####
        # keep loop time constant
        clock.tick(CONST.SCREEN_FPS)
        screen.fill(CONST.COLOR_BLACK)
    
        # Returns quality estimates for all posiable actions
        qMatrix = dqnn.getQMat(state.state.flatten())
        state_0_flat = state.state.flatten()
        
        ##### SELECT ACTION #####
        # select random action or use best action from qMatrix
        action_idx = 0
        if (random.random() < epsilon):
            action_idx = random.randint(0,len(CONST.ACTION_AND_COSTS)-1)
            #print("Random action!!!")
        else:
            action_idx = np.argmax(qMatrix)
#        print(CONST.ACTION_NAMES[action_idx])
#        print("speed: ", car.speed)
        #print("Action: ", CONST.ACTION_NAMES[action_idx], "-- --Epoch: ", epoch_cnt, "-- --Episode: ", episode, '-- --Epslion: ', epsilon)
        ##### Take action #####
        #print("Action: {0}".format(CONST.ACTION_AND_COSTS[action_idx]))
        car.updateAction(action_idx, force  = True if len(replay) < buffer else False)   # apply action selected above
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
        
        # Check for additional penalties from dangerous driving        
        if car.tail_gaiting:
            reward += CONST.REWARDS['tail_gate']
            #print('tail_gate')
        
        if car.lane_idx == 0:
            reward += CONST.REWARDS['on_sholder']
            #print('on_sholder')
            
        if car.speed < CONST.MIN_SPEED:
            reward += CONST.REWARDS['too_slow']
            #print('too_slow')
         # Check for terminal states and override 
        # reward to teminal values if necessary
        if (collisions or (car.lane_idx < 0) or (car.lane_idx > 3) or (ticks > CONST.TIME_TO_GIVE_UP)):
            pigs_fly = True
            reward = CONST.REWARDS['terminal_crash']
            #print('terminal_crash')
        
        if car.isAtGoal():
            pigs_fly = True
            reward = CONST.REWARDS['terminal_goal']
            car.terminal = True
            #print('terminal_goal')
        #print("Reward: ", reward)
        
        if len(replay) < buffer:
            replay.append((state_0_flat, action_idx, reward, state.state))
            epoch_cnt =  0# keep epochs at zero until buffer if full
        else:
            CONST.SCREEN_FPS = 20
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
                #q_val_old = np.argmax(q_mat_old)
                y =  q_mat_old[:]

                q_mat_new = dqnn.getQMat(new_state.flatten())
                q_val_new = np.argmax(q_mat_new)
        
                target_q = reward + (gamma*q_val_new)
                y[0][action_idx] = target_q
                target_batch.append(y)
                
            if epoch_cnt > 0 and epoch_cnt % 20 == 0:   
                dqnn.fitBatch([row[0] for row in batch], target_batch, save=False, idx=action, update=target_q)
                print("epoch: {0}, epsilon: {1}".format(epoch_cnt, epsilon))
                
            if epoch_cnt > 0 and epoch_cnt % 100 == 0 and save:
                save = False
                dqnn.fitBatch([row[0] for row in batch], target_batch, save=True)
        
            if epsilon > 0.1:
                epsilon -= 1/epochs
        
        doObsMerge(merge_count)
        
#        if epoch_cnt > 0:
#            print("Epochs: ", epoch_cnt)
#            print("target_q: {0} = reward: {1} + gamma:{2} * (qMax: {3})".format(target_q, reward, gamma, qMax))
#            print("action_idx: {0}".format(action_idx))
#            print("Episilon: ", epsilon)
#            print("Q_mat: ", qMatrix)   
            
            
        ##### MORE PYGAME HOUSE KEEPING #####
    #   respawn obstacles if they are out of range
        if moving_obstacles:
            for obs in obstacles:            
                if obs.out_of_range:
                    obs.reInitObs(0, CONST.LANES[random.rand(CONST.CAR_LANE_MIN,CONST.CAR_LANE_MAX)], obstacles)
                    #print("Reload")
                
                
#       check if car is out of bounds
        if car.out_of_bounds:
            pigs_fly = True
            #print("Out_Of_bounds")
                
    #    Draw / render
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
#        if episode % 100 == 0:
#                print("episode: {0}".format(episode))
    save = True
    epoch_cnt += 1    
    if leave_program: break

dqnn.session.close()
pygame.quit();
