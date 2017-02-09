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
import data_logging as log

log_data = True
load_data = False
assert not (log_data and load_data), "Cannot log and load at the same time."
fileNames = {"states0" : "states0_{0}.txt".format("STAGE2_180Lidar"),
             "actions" : "actions_{0}.txt".format("STAGE2_180Lidar"),
             "rewards" : "rewards_{0}.txt".format("STAGE2_180Lidar"),
             "states1" : "states1_{0}.txt".format("STAGE2_180Lidar")}
states0 = []
rewards = []
actions = []
states1 = []
toLog = [states0, rewards, actions, states1]
if log_data:
    log.logDataInit(fileNames)

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



def initSimulation(car, state, filling_buffer = False, x_dist = 0, lane=2):

    global obstacels
    global all_sprites
    obstacles.empty()
    all_sprites.empty()
    all_sprites.add(car)

    # This is a super hacky way to try and bias away from initialising the car in 'breaking' state
    # when filling replay buffer
    
#    random_start = random.randint(0,len(CONST.ACTION_NAMES)-1)
#    random_start = random.randint(0,len(CONST.ACTION_NAMES)-1) if random_start == 3 else random_start
    obs_x = [200]
    obs_lanes = [2]
#    rand_x = random.uniform(0, CONST.SCREEN_WIDTH*0.75)
#    rand_y = random.randint(1,3)
#
#    valid = False
#    while not valid:
#        valid = not ((rand_y == 1 and abs(rand_x - obs_x[1-1]) < 1.5*CONST.CAR_SAFE_BUBBLE) or
#                     (rand_y == 2 and abs(rand_x - obs_x[2-1]) < 1.5*CONST.CAR_SAFE_BUBBLE) or
#                     (rand_y == 3 and abs(rand_x - obs_x[3-1]) < 1.5*CONST.CAR_SAFE_BUBBLE))
#        rand_x = random.uniform(0, CONST.SCREEN_WIDTH*0.75)
#        rand_y = random.randint(1,3)
#        
#    if filling_buffer:
#        car.reInit(rand_x, rand_y)
#    else:
    lane = 2
    x_dist = 0
    if filling_buffer:
        lane = random.randint(1,3)
        x_dist = int(random.uniform(0,0.5)*CONST.SCREEN_WIDTH)
        while ((lane == 2) and abs(x_dist - obs_x[0]) < CONST.CAR_SAFE_BUBBLE):
            x_dist = int(random.uniform(0,0.5)*CONST.SCREEN_WIDTH)

    car.reInit(x_dist, lane)

    #initObstacles(CONST.OBS_L_TO_R,CONST.OBS_R_TO_L)
    initStaticObstacles(xPos=obs_x, lanes=obs_lanes)

# Initial state assumes the agent
# Has been stationary for a short period
    car.updateSensors(obstacles)


    ########## I'M GOING WHERE THE ACTION ISSSS!!!!! ################
total_frames = 0
epochs = 50000
epoch_cnt = 0
gamma = 0.9
epsilon = 1
leave_program = False
batch_size = 30
buffer = 30000
replay = []
h = 0
reward = 0
qMax = 0

for i in range(epochs):

    initSimulation(car, state, filling_buffer = True if len(replay) < buffer else False)
    pigs_fly = False
    frames_this_epoch = 0

    while not pigs_fly:
        print("FRAME: {0}".format(frames_this_epoch))
#####  PYGAME HOUSE KEEPING  #####
# Keep loop time constant
        clock.tick(CONST.SCREEN_FPS)
        screen.fill(CONST.COLOR_BLACK)

# Returns quality estimates for all posiable actions
        qMatrix = dqnn.getQMat(state.state)
        state_0 = copy.deepcopy(state.state)
        print("q_matrix: ", qMatrix)
##### SELECT ACTION #####
# Select random action or use best action from qMatrix
        action_idx = 0
        if (random.random() < epsilon):
            action_idx = random.randint(0,len(CONST.ACTION_AND_COSTS)-1)
            print("random action: ", CONST.ACTION_NAMES[action_idx])
        else:
            action_idx = np.argmax(qMatrix)
            print("selected action: ", CONST.ACTION_NAMES[action_idx])

##### Take action #####
#        car.updateAction(2)   # apply action selected above
        car.updateAction(action_idx)   # apply action selected above
        all_sprites.update()                                               # Pygame updating sprites
        collisions = pygame.sprite.spritecollide(car, obstacles, False)    # Check for agent obstacle collisions

##### Observe new state (s') #####
        car.updateSensors(obstacles)   # Sensor update
        state.update(car.sensor_data)  # Update state with new data

##### GET maxQ' from DCNN #####
        next_qMatrix = dqnn.getQMat(state.state)

        reward = car.reward
# Check for terminal states and override
# Reward to teminal values if necessary
#        if car.tail_gaiting:
#            reward = CONST.REWARDS['tail_gate']
#            pigs_fly = True

#        if car.lane_idx == 0:
#            reward = CONST.REWARDS['on_sholder']
#            pigs_fly = True
            
#        if car.speed < CONST.MIN_SPEED:
#            reward = CONST.REWARDS['too_slow']
#            pigs_fly = True

        if (collisions or car.out_of_bounds):
            pigs_fly = True
            reward = CONST.REWARDS['terminal_crash']
            print("terminal_crash*********************************************************************************")
            
        if car.isAtGoal():
            pigs_fly = True
            reward = CONST.REWARDS['terminal_goal']
            car.terminal = True
            print("terminal_goal!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            
        if frames_this_epoch > CONST.TAKING_TOO_LONG:
            pigs_fly = True
            print("RESET!!!!! TAKING TOO LONG!!!!!!!!!!!!!!!!!")
        print("reward: ", reward)
        if len(replay) < buffer:
            replay.append((copy.deepcopy(state_0), copy.deepcopy(action_idx), copy.deepcopy(reward), copy.deepcopy(state.state)))
        
            if log_data:
                # Data Logging    
                states0.append(copy.deepcopy(state_0))
                actions.append(copy.deepcopy(action_idx))
                rewards.append(copy.deepcopy(reward))
                states1.append(copy.deepcopy(state.state))

            epoch_cnt =  0# keep epochs at zero until buffer if full
        else:
            CONST.SCREEN_FPS = 50
            h = (h+1)%buffer

            replay.append((copy.deepcopy(state_0), copy.deepcopy(action_idx), copy.deepcopy(reward), copy.deepcopy(state.state)))
#            print("action: {0}, reward: {1}, ".format(action_idx, reward))
            batch = random.sample(replay, batch_size)
            target_batch = []
            for element in batch:
                replay_old_state, replay_action_idx, replay_reward, replay_new_state = element
                q_mat_old = dqnn.getQMat(replay_old_state)

                y =  copy.deepcopy(q_mat_old[0])

                q_mat_new = dqnn.getQMat(replay_new_state)[0]
                q_val_new = max(q_mat_new)
                if (replay_reward == -10 or 
                    replay_reward == -5 or
                    replay_reward == 10):
                    q_update = replay_reward
                else:
                    q_update = replay_reward + (gamma*q_val_new)
                y[replay_action_idx] = q_update
                if total_frames % 100 == 0:
                    print("Y: {0}, idx: {1}, target: {2} (ori: {3}), epsilon: {4}".format(y, replay_action_idx, q_update, q_val_new, epsilon))
                    print("epoch_cnt: ", epoch_cnt)
                    
                target_batch.append(y)


            if total_frames % 100 == 0:
                dqnn.fitBatch([row[0] for row in batch], target_batch, save=False, verbose=True, iteration_count=total_frames-buffer)
            elif total_frames % 1001 == 0:
                dqnn.fitBatch([row[0] for row in batch], target_batch, save=True, verbose=False, iteration_count=total_frames-buffer)
            else:
                dqnn.fitBatch([row[0] for row in batch], target_batch, save=False)

            if epsilon > 0.1:
                epsilon -= 1/epochs

#        doObsMerge(merge_count)

##### MORE PYGAME HOUSE KEEPING #####
# Respawn obstacles if they are - CONST. out of range
        if moving_obstacles:
            for obs in obstacles:
                if obs.out_of_range:
                    obs.reInitObs(0, CONST.LANES[random.rand(CONST.CAR_LANE_MIN,CONST.CAR_LANE_MAX)], obstacles)
                    #print("Reload")
            
# Check if car is out of bounds
        if car.rect.x > CONST.SCREEN_WIDTH + CONST.SCREEN_PADDING:
            pigs_fly = True

# Draw / render
        all_sprites.draw(screen)
### Drawing lane markers
        center_guard = CONST.LANES[3] + CONST.LANE_WIDTH//2
        color = CONST.COLOR_ORANGE
        for lane in CONST.LANES:
            pygame.draw.line(screen, color, (0, lane-CONST.LANE_WIDTH//2), (CONST.SCREEN_WIDTH,  lane-CONST.LANE_WIDTH//2))
            color = CONST.COLOR_WHITE
        pygame.draw.line(screen, CONST.COLOR_ORANGE, (0, CONST.LANES[len(CONST.LANES)-1] + CONST.LANE_WIDTH//2), (CONST.SCREEN_WIDTH,  CONST.LANES[len(CONST.LANES)-1] + CONST.LANE_WIDTH//2))

## Draw carrot (what the PID follows track lanes)
        pygame.draw.circle(screen, CONST.COLOR_ORANGE, (car.carrot), 5)
        pygame.draw.circle(screen, CONST.COLOR_ORANGE, (300, int(CONST.LANES[3] + CONST.LANE_WIDTH//2)), 4)
#
## Draw most recent LiD
        for beam in car.lidar.beams:
            pygame.draw.line(screen, beam.color, (beam.x1, beam.y1), (car.rect.centerx, car.rect.centery))

##### For if I decide to put some user input fuctionality  #####
# Process input (events)
        for event in pygame.event.get():
# Check for closing window
            if event.type == pygame.QUIT:
                pigs_fly = True
                leave_program = True
                dqnn.session.close()

        frames_this_epoch += 1
        total_frames += 1
        if total_frames % 100 == 0:
            print("total_frames: ", total_frames)
# After everything, flip display
        pygame.display.flip()



    epoch_cnt += 1
    if epoch_cnt == epochs-1: epoch_cnt = epochs - 2 
    if leave_program: break
    if log_data:
        log.logData(fileNames, toLog)
        # Data Logging    
        states0.clear()
        actions.clear()
        rewards.clear()
        states1.clear()

dqnn.session.close()
pygame.quit();
