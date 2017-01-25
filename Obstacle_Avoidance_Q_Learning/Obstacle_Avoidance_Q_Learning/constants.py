# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 16:20:00 2016

@author: Josh
"""
import math

#colors
COLOR_RED = (255,0,0)
COLOR_GREEN = (0,255,0)
COLOR_BLUE = (0,0,255)
COLOR_BLACK = (0,0,0)
COLOR_WHITE = (255,255,255)
COLOR_ORANGE = (255,102,0)
COLOR_YELLOW = (255,255,0)

CAR_LENGTH = 44 #length in pixles
CAR_SAFE_BUBBLE = math.ceil(2*CAR_LENGTH)
CELLS_PER_LANE = 11

#screen
SCREEN_WIDTH = math.ceil(CELLS_PER_LANE*CAR_SAFE_BUBBLE)  #keep in multaples of car_safe_bubble so to make occupancy grid uniform
SCREEN_HEIGHT = 400
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
SCREEN_PADDING = 200
SCREEN_FPS = 30

ONE_DEGREE = 3.14159 / 180.0
CAR_ANGULAR_ACCEL = ONE_DEGREE * 1
CAR_FORWARD_ACCEL = 0.5

LANE_WIDTH = 40
LANES = [SCREEN_HEIGHT//2 - (2*LANE_WIDTH),
         SCREEN_HEIGHT//2 - (1*LANE_WIDTH),
         SCREEN_HEIGHT//2 - (0*LANE_WIDTH),
         SCREEN_HEIGHT//2 + (1*LANE_WIDTH),
         SCREEN_HEIGHT//2 + (2*LANE_WIDTH),
         SCREEN_HEIGHT//2 + (3*LANE_WIDTH)]
DIRECTIONS = {"l_to_r": 0,
              "r_to_l": math.pi}
              

INIT_SPEED = 10
MAX_SPEED = 20
MIN_SPEED = 3
OBS_L_TO_R = 5
OBS_R_TO_L = 6
MERGE_PROB = 0.20
MAX_NO_MERGES = 4

LIDAR_RANGE =  200
LIDAR_COUNT = 50
LIDAR_SWEEP = 220
LIDAR_RES = 5 # one pixle is approx 10cm
LIDAR_STEP = LIDAR_SWEEP / (LIDAR_COUNT - 1)
LIDAR_DATA_SIZE = (LIDAR_COUNT, (LIDAR_RANGE // LIDAR_RES))

HISTORY_DEPTH = 4
STATE_MATRIX_SIZE = (HISTORY_DEPTH, LIDAR_COUNT, (LIDAR_RANGE // LIDAR_RES))
#
#ACTION_AND_COSTS = [('do_nothing',           0),
#                    ('soft_left',           -1),
#                    ('medium_left',         -3),
#                    ('hard_left',           -5),
#                    ('soft_right',          -1),
#                    ('medium_right',        -3),
#                    ('hard_right',          -5),
#                    ('soft_break',          -1),
#                    ('medium_break',        -3),
#                    ('hard_break',          -5),
#                    ('soft_acceleration',   -1),
#                    ('medium_acceleration', -3),
#                    ('hard_acceleration',   -5)]


ACTION_AND_COSTS = [('do_nothing',           0),
                    ('change_left',         -1),
                    ('change_right',        -1),
                    ('break',               -1),
                    ('accelerate',          -1)]
           

URGENCY = {0: 'out_of_range',
          1: 'safe',
          2: 'uneasy',
          3: 'dangerous',
          4: 'emergency',
          5: 'terminal_goal',
          6: 'terminal_crash'}

REWARDS =           {URGENCY[0] :  0,  #o_o_r
                     URGENCY[1] :  0,  #safe
                     URGENCY[2] :  0,  #unease
                     URGENCY[3] :  0,  #dangerous
                     URGENCY[4] :  0, #emergency
                     URGENCY[5] :   10, #goal
                     URGENCY[6] :  -10} #crash

LIDAR_COLORS =      {URGENCY[0] : COLOR_BLUE,
                     URGENCY[1] : COLOR_GREEN,
                     URGENCY[2] : COLOR_YELLOW,
                     URGENCY[3] : COLOR_ORANGE,
                     URGENCY[4] : COLOR_RED}

