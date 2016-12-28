# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:20:00 2016

@author: Josh
"""

#colors
COLOR_RED = (255,0,0)
COLOR_GREEN = (0,255,0)
COLOR_BLUE = (0,0,255)
COLOR_BLACK = (0,0,0)
COLOR_WHITE = (255,255,255)
COLOR_ORANGE = (255,102,0)
COLOR_YELLOW = (255,255,0)

#screen
SCREEN_WIDTH = 1400
SCREEN_HEIGHT = 700
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
SCREEN_FPS = 60


ONE_DEGREE = 3.14159 / 180.0
CAR_ANGULAR_ACCEL = ONE_DEGREE * 1
CAR_FORWARD_ACCEL = 0.5
CAR_MAX_SPEED = 5

OBSTACLE_MAX_SPEED = 10
OBSTACLE_MIN_SPEED = 3

URGENCY = {0: 'out_of_range',
          1: 'safe',
          2: 'uneasy',
          3: 'dagerous',
          4: 'emergency',
          5: 'terminal_goal',
          6: 'terminal_crash'}

REWARDS =           {URGENCY[0] :   0,
                     URGENCY[1] :  -1,
                     URGENCY[2] :  -2,
                     URGENCY[3] :  -5,
                     URGENCY[4] :  -7,
                     URGENCY[5] :  10,
                     URGENCY[6] : -10}

LIDAR_COLORS =      {URGENCY[0] : COLOR_BLUE,
                     URGENCY[1] : COLOR_GREEN,
                     URGENCY[2] : COLOR_YELLOW,
                     URGENCY[3] : COLOR_ORANGE,
                     URGENCY[4] : COLOR_RED}

