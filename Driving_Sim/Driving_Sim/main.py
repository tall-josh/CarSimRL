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

#setting up lidar_history
lidar_history = []
history_depth = 5 # keep 5 frames of past lidar data
for i in range(history_depth):
    lidar_history.append([])
history_idx = 0   # current frame being written to

#init obstacles
for i in range(random.randint(1,5)):
    obs = obstacle.Obstacle(art.cars['gray'],car.rect.centerx, car.rect.centery, CONST.LIDAR_RANGE)
    all_sprites.add(obs)
    obstacles.add(obs)
                 
# Game Loop
score = 0
count  = 0
running = True
while running:
    # keep loop time constant
    clock.tick(CONST.SCREEN_FPS)
    screen.fill(CONST.COLOR_BLACK)
    
    # Process input (events)
    for event in pygame.event.get():
        #check for closing window
        if event.type == pygame.QUIT:
            running = False
     
    keystate = pygame.key.get_pressed()
       
    # Update
    all_sprites.update()
        
    # Check for agent obstacle collisions
    collisions = pygame.sprite.spritecollide(car, obstacles, False)
    
    # sensor update        
    lidar_data = car.updateSensors(obstacles) 
    
    #update sensor history
    lidar_history[history_idx] = lidar_data
    history_idx += 1
    if history_idx == history_depth:
        history_idx = 0
    
    
    for obs in obstacles:
        if obs.out_of_range:
            obs.initState(car.rect.centerx, car.rect.centery, CONST.LIDAR_RANGE)
        if obs.tag:
            count += 1
            obs.tag = False
            
#    Draw / render
    all_sprites.draw(screen)
    
#   Draw goal
    pygame.draw.circle(screen, CONST.COLOR_WHITE, (car.goal), 10) 
    
#   DRAW MOST RECENT Lidar
    for beam in car.lidar.beams:
        pygame.draw.line(screen, beam.color, (beam.x1, beam.y1), (car.rect.centerx, car.rect.centery))

#    DRAW ENTIRE Lidar HISTORY
#    for ele in lidar_history:
#        for hit in ele:
#            pygame.draw.line(screen, hit[2], (hit[0], hit[1]), (car.rect.centerx, car.rect.centery))
    
    if collisions:
        running = False
        score += CONST.REWARDS["terminal_crash"]
        print("Terminal Reward (COLLISION): {0}".format(CONST.REWARDS["terminal_crash"]))
    if car.at_goal:
        running = False
        score += CONST.REWARDS["terminal_goal"]
        print("Terminal Reward (AT GOAL): {0}".format(CONST.REWARDS["terminal_goal"]))

    print("Count: {0}, Closest {1}, Lidar_Reward: {2}".format(count, car.lidar.closest_dist, car.lidar.reward))
    score += car.lidar.reward
    print("Score: {0}".format(score))

    # After everything, flip display
    pygame.display.flip()
pygame.quit();
