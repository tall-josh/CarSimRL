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
obstacles = pygame.sprite.Group()
car = car.Car(art.cars['player'])
all_sprites.add(car)

beam_length = 500
no_of_beams = 100
lidar = lidar.Lidar(beam_length, no_of_beams, 220, car.rect.centerx, car.rect.centery, math.degrees(car.heading), 10)
car.attachLidar(lidar)

#init obstacles
for i in range(random.randint(1,5)):
    obs = obstacle.Obstacle(art.cars['gray'],car.rect.centerx, car.rect.centery, beam_length)
    all_sprites.add(obs)
    obstacles.add(obs)

#setting up lidar_history
lidar_history = []
history_depth = 5 # keep 5 frames of past lidar data
for i in range(history_depth):
    lidar_history.append([])
    
history_idx = 0   # current frame being written to
                 
# Game Loop
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

    # Update
    all_sprites.update()
        
    # Collisions
    collisions = pygame.sprite.spritecollide(car, obstacles, False)
    
    # sensor update
    for obs in obstacles:
        obs.resetUrgency()
        
    lidar_hits = car.updateSensors(obstacles) 
    
    reward = 0
    for obs in obstacles:
        reward += CONST.REWARDS[CONST.URGENCY[obs.urgency]]
        print('urgency: {0}'.format(CONST.URGENCY[obs.urgency]))
        if obs.out_of_range:
            obs.initHeading(car.rect.centerx, car.rect.centery, beam_length)
    print('reward: {0}'.format(reward))
    
    #update history
    lidar_history[history_idx] = lidar_hits
    history_idx += 1
    if history_idx == history_depth:
        history_idx = 0
        
    if collisions or car.at_goal:
        running = False
    
    # Draw / render
        #Sprites
    all_sprites.draw(screen)
    
##   Check if obstacles are out of range
#    print('line')
#    reward = 0
#    for obs in obstacles:
#        print('urgency: {0}'.format(CONST.URGENCY[obs.urgency]))
#        reward += CONST.REWARDS[CONST.URGENCY[obs.urgency]]
#        if obs.out_of_range:
#            obs.initHeading(car.rect.centerx, car.rect.centery, beam_length)
#    print('reward: {0}'.format(reward))
#   Draw goal
    pygame.draw.circle(screen, CONST.COLOR_WHITE, (car.goal), 10) 
    
#   DRAW MOST RECENT Lidar
#   lidar_hits is an array of [x, y, 'state'].
    for hit in lidar_hits:
        pygame.draw.line(screen, hit[2], (hit[0], hit[1]), (car.rect.centerx, car.rect.centery))
    
#    DRAW ENTIRE Lidar HISTORY
#    for ele in lidar_history:
#        for hit in ele:
#            pygame.draw.line(screen, hit[2], (hit[0], hit[1]), (car.rect.centerx, car.rect.centery))
            
    # After everything, flip display
    pygame.display.flip()

pygame.quit();