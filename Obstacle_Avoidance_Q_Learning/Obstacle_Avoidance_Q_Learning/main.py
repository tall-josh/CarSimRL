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

# Create Agent and attach sensors
car = car.Car()

lidar = lidar.Lidar(car.rect.centerx, car.rect.centery, math.degrees(car.heading))
car.attachLidar(lidar)
all_sprites.add(car)

# initalise 
state = st.StateTracker(CONST.STATE_MATRIX_SIZE[0], CONST.STATE_MATRIX_SIZE[1], CONST.STATE_MATRIX_SIZE[2])
    
# super sloppy repeting code like this soz !soz.
def initSimulation(car, state):
    
    global obstacels
    global all_sprites
    obstacles.empty()
    all_sprites.empty()
    all_sprites.add(car)
    
    car.reInit()
    state.reset()
    
    #init obstacles
    for i in range(random.randint(1,CONST.MAX_NUMBER_OF_OBSTACLES)):
        obs = obstacle.Obstacle(art.cars['gray'],car.rect.centerx, car.rect.centery, CONST.LIDAR_RANGE)
        all_sprites.add(obs)
        obstacles.add(obs)
        
    #initial state assumes the agent 
    #has been stationary for a short period
    car.updateSensors(obstacles)
    state.update(car.sensor_data)
    
    
    ########## I'M GOING WHERE THE ACTION ISSSS!!!!! ################
score = 0               # total score of the round
ticks  = 0              # number of iterations in each round will give up if > than...
max_ticks_before_we_just_give_up = 100
epochs = 10
gamma = 0.9
epsilon = 1
leave_program = False
batch_size = 40
buffer = 80
replay = []

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
        
        ##### SELECT ACTION #####
        # select random action or use best action from qMatrix
        action_idx = 0
##### UNCOMMENT WHEN YOU"RE DONE CHECKING CONTROLS OF CAR ######
#        if (random.random() < epsilon):
#            action_idx = random.randint(0,11)
#        else:
#            action_idx = np.argmax(qMatrix)
       
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
        qMax = np.argmax(next_qMatrix)
        
        ##### Generate target vector to train network #####
        target = copy.deepcopy(qMatrix)
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
            print("Terminal Goal")
        
        target_q = reward + (gamma*qMax)
    
        # override quality predicted by DNN for this action
        # to target_q calculated above. 
        target[0][action_idx] = target_q
        
        dqnn.fit(state.state.flatten(), target.flatten())
        
        ##### MORE PYGAME HOUSE KEEPING #####
    #   respawn obstacles if they are out of range
        for obs in obstacles:            
            if obs.out_of_range:
                obs.initState(car.rect.centerx, car.rect.centery, CONST.LIDAR_RANGE)
                print("Reload")
                
#       check if car is out of bounds
        if car.out_of_bounds or (ticks == max_ticks_before_we_just_give_up):
            pigs_fly = True
            print("Out_Of_bounds")
                
    #    Draw / render
        all_sprites.draw(screen)
        
    #   Draw goal
        pygame.draw.circle(screen, CONST.COLOR_WHITE, (car.goal), 10) 
        
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
