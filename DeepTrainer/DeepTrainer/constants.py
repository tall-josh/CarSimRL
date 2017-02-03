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
CELLS_PER_LANE = 6

#screen
SCREEN_WIDTH = math.ceil(CELLS_PER_LANE*CAR_SAFE_BUBBLE)  #keep in multaples of car_safe_bubble so to make occupancy grid uniform
SCREEN_HEIGHT = 400
SCREEN_SIZE = (SCREEN_WIDTH, SCREEN_HEIGHT)
SCREEN_PADDING = 200
SCREEN_FPS = 15

ONE_DEGREE = 3.14159 / 180.0
CAR_ANGULAR_ACCEL = ONE_DEGREE * 1
CAR_FORWARD_ACCEL = 1.0
LANE_WIDTH = 40
SHOLDER = LANE_WIDTH//2

__offset = 50
LANES = [(__offset + 0*LANE_WIDTH),
         (__offset + 1*LANE_WIDTH),
         (__offset + 2*LANE_WIDTH),
         (__offset + 3*LANE_WIDTH),
         (__offset + 4*LANE_WIDTH),
         (__offset + 5*LANE_WIDTH),
         (__offset + 6*LANE_WIDTH),
         (__offset + 7*LANE_WIDTH)]
OBSTICALE_LANES = LANES[1:len(LANES)-1]

__increment = 10
__overshoot = 50*__increment
__no_of_increments = int((((LANES[len(LANES)-1] + __overshoot) - (LANES[0] - __overshoot)) / __increment) + 1)
DRIVING_LANES = [LANES[0]-__overshoot+(__increment*i) for i in range(__no_of_increments)]

CAR_LANE_MIN = 1 #the car can drive on the sholder, but it cannot be initalized there
CAR_LANE_MAX = 3
OBS_LN_LtoR_MIN = CAR_LANE_MIN
OBS_LN_LtoR_MAX = CAR_LANE_MAX
OBS_LN_RtoL_MIN = CAR_LANE_MAX + 1
OBS_LN_RtoL_MAX = len(LANES) - 2 # minus 2 because the obs cannot drive on the sholder

         
DIRECTIONS = {"l_to_r": 0,
              "r_to_l": math.pi}
              

INIT_SPEED = 8
MAX_SPEED = 11
MIN_SPEED = 5
OBS_L_TO_R = 5
OBS_R_TO_L = 6
MERGE_PROB = 0
MAX_NO_MERGES = 4

LIDAR_RANGE =  200
LIDAR_COUNT = 21
LIDAR_SWEEP = 359
LIDAR_RES = 3 # one pixle is approx 10cm
LIDAR_STEP = LIDAR_SWEEP / (LIDAR_COUNT - 1)
LIDAR_DATA_SIZE = (LIDAR_COUNT, (LIDAR_RANGE // LIDAR_RES))
STATE_MATRIX_SIZE = LIDAR_DATA_SIZE
STATE_MATRIX_FLAT_SZ = STATE_MATRIX_SIZE[0]*STATE_MATRIX_SIZE[1]
HISTORY_WEIGHTS = [0.125, 0.25, 0.5, 1] #oldest to newest
HISTORY_DEPTH = 4
FRAME_HISTORY_SIZE = (HISTORY_DEPTH, LIDAR_COUNT, (LIDAR_RANGE // LIDAR_RES))
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

CAR_CONTROL_DAMPENING_DEPTH = 3

ACTION_NAMES = ['do_nothing',
                'change_left',
                'change_right',
                'break',
                'accelerate']


ACTION_AND_COSTS = [('do_nothing',           0),
                    ('change_left',         -1),
                    ('change_right',        -1),
                    ('break',               -1),
                    ('accelerate',          -1)]
           
TAIL_GATE_DIST = LIDAR_RANGE*0.3                   
REWARDS =           {'terminal_crash' :    -10,  
                     'terminal_goal'  :     10,  
                     'on_sholder'     :    -5,  
                     'tail_gate'      :    -5,
                     'too_slow'       :    -5}
