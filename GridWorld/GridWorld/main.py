# -*- coding: utf-8 -*-
"""
Created on Sat Dec 10 13:42:26 2016

@author: Josh
"""
'''
// Definitions in C# syntax

public Action Policy(State s){
    // using the given State 's', return the best Action;
    // A Policy is usually denoted in literature as Pi(s)
}

// ValueFunction or ActionValueFunction
// Some RL may use either
// Formally: 'the value of a State is the 
// value of the highest action taken from that State.'

public float ValueFunction(State s){
    // returns the 'value' of the given State 's'
}

--> OR

public float ActionValueFunction(State s, Action a){
    // Returns value of taking Action 'a' given State 's'
}

// For toy application these functions could be a 
// look-up table, but, fuck that for a joke. We're
// going to train an ANN to do this for us :-)
// The ANN will need to take an additional vetor of
// parameters it will update so it can learn more
// accurate State-Action values.

public float ActionValueFunctionANN(State s, Action a, Params theta){
    
}

//

'''

import numpy as np

#return random integer pair between Start 's' and End 'e'
def randPair(s, e):
    return np.random.randint(s,e), np.random.randint(s,e)

#finds an array in the 'depth' dimention of the grid
#'state' is the grid borad 
def findLoc(state, obj):
    for i in range(0,4):
        for j in range (0,4):
            if (state[i,j] == obj).all():
                return i,j
                
#Initalize stationary grid, all items place deterministacally
def initGrid():
    state = np.zeros((4,4,4))
    #place player
    state[0,1] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[3,3] = np.array([1,0,0,0])
    
    return state

#Initalize player in random location
def initGridPlayer():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[2,2] = np.array([0,0,1,0])
    #place pit
    state[1,1] = np.array([0,1,0,0])
    #place goal
    state[3,3] = np.array([1,0,0,0])
    
    #find grid position of player (agent), p is for pit
    a = findLoc(state, np.array([0,0,0,1]))
    #find wall
    w = findLoc(state, np.array([0,0,1,0]))
    #find goal
    g = findLoc(state, np.array([1,0,0,0]))
    #find pit
    p = findLoc(state, np.array([0,1,0,0]))
    
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return initGridPlayer()
    
    return state

#Initalize grid so that goal, pit, wall and player
#all randomly located
def initGridRand():
    state = np.zeros((4,4,4))
    #place player
    state[randPair(0,4)] = np.array([0,0,0,1])
    #place wall
    state[randPair(0,4)] = np.array([0,0,1,0])
    #place pit
    state[randPair(0,4)] = np.array([0,1,0,0])
    #place goal
    state[randPair(0,4)] = np.array([1,0,0,0])

    a = findLoc(state, np.array([0,0,0,1]))
    w = findLoc(state, np.array([0,0,1,0]))
    g = findLoc(state, np.array([1,0,0,0]))
    p = findLoc(state, np.array([0,1,0,0]))
    #If any of the "objects" are superimposed, just
    #call  the function again to re-place
    if (not a or not w or not g or not p):
        #print('Invalid grid. Rebuilding..')
        return initGridRand()

    return state


def makeMove(state, action):
    #need to locate player in grid
    #need to determine what object (if any) is in the new grid spot the player is moving to
    player_loc = findLoc(state, np.array([0,0,0,1]))
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    state = np.zeros((4,4,4))

    #up (row - 1)
    if action==0:
        new_loc = (player_loc[0] - 1, player_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #down (row + 1)
    elif action==1:
        new_loc = (player_loc[0] + 1, player_loc[1])
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #left (column - 1)
    elif action==2:
        new_loc = (player_loc[0], player_loc[1] - 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1
    #right (column + 1)
    elif action==3:
        new_loc = (player_loc[0], player_loc[1] + 1)
        if (new_loc != wall):
            if ((np.array(new_loc) <= (3,3)).all() and (np.array(new_loc) >= (0,0)).all()):
                state[new_loc][3] = 1

    new_player_loc = findLoc(state, np.array([0,0,0,1]))
    if (not new_player_loc):
        state[player_loc] = np.array([0,0,0,1])
    #re-place pit
    state[pit][1] = 1
    #re-place wall
    state[wall][2] = 1
    #re-place goal
    state[goal][0] = 1

    return state

def getLoc(state, level):
    for i in range(0,4):
        for j in range(0,4):
            if (state[i,j][level] == 1):
                return i,j
                
def getReward(state):
    player_loc = getLoc(state, 3)
    pit = getLoc(state, 1)
    goal = getLoc(state, 0)
    if(player_loc == pit):
        return -10
    elif(player_loc == goal):
        return 10
    else:
        return -1
        
    
def dispGrid(state):
    grid = np.zeros((4,4),dtype='<U2')
    player_loc = findLoc(state, np.array([0,0,0,1]))        
    wall = findLoc(state, np.array([0,0,1,0]))
    goal = findLoc(state, np.array([1,0,0,0]))
    pit = findLoc(state, np.array([0,1,0,0]))
    
    for i in range(0,4):
        for j in range(0,4):
            grid[i,j] = ' '
    
    if player_loc:
        grid[player_loc] = 'P'
    if wall:
        grid[wall] = 'W'
    if goal:
        grid[goal] = 'G'
    if pit:
        grid[pit] = '-'    

    return grid
    

state = initGridRand()
print("MAP")
print(dispGrid(state))

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import RMSprop

model = Sequential()
model.add(Dense(164, init='lecun_uniform', input_shape=(64,)))
model.add(Activation('relu'))
#model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a go?

model.add(Dense(150, init='lecun_uniform'))
model.add(Activation('relu'))
#model.add(Dropout(0.2))

model.add(Dense(4, init='lecun_uniform'))
model.add(Activation('linear')) #linear output so we can have range of real-valued outputs

rms = RMSprop()
model.compile(loss='mse', optimizer=rms)

#print("Pridict:")
#print(model.predict(state.reshape(1,64), batch_size=1))
#just to show an example output; read outputs left to right: up/down/left/right
        
from IPython.display import clear_output
import random

model.compile(loss='mse', optimizer=rms) #reset weights of neural network
epochs = 50000
gamma = 0.975 #since it may take several moves to goal, making gmma high
epsilon = 1
batchSize = 40
buffer = 80
replay = [] #stores tuples of (S, A, R, S')
h = 0

for i in range(epochs):
    
    if i%10 == 0:
        print ('Epoch: %i of %i' % (i,epochs))
        
    state = initGridRand() #hardest initalisation state
    #state = initGridPlayer() #harder initalisation state
    #state = initGrid() #easier initislation state
    status = 1
    #while game still in progress
    while(status == 1):
        
        #we are in state S
        #Lets run our Q function on S to get Q values for all possiable actions
        qval = model.predict(state.reshape(1,64), batch_size=1)
        if (random.random() < epsilon): #choose random action
            action = np.random.randint(0,4)
        else: #choose best action from Q(S,a) values
            action = (np.argmax(qval))
        #take action, observe new state S'
        new_state = makeMove(state, action)
        #observe reward
        reward = getReward(new_state)
        #get max_Q(s', a)
        
        #EXPERIENCE REPLAY STORAGE
        if (len(replay) < buffer): #if buffer not full, add to it
            replay.append((state, action, reward, new_state))
        else: #if buffer full, overwrite old vals
            if(h < (buffer - 1)):
                h += 1
            else:
                h = 0
            replay[h] = (state, action, reward, new_state)
            #randomly sample our experience replay memory
            minibatch = random.sample(replay, batchSize)
            X_train = []
            Y_train = []
            for memory in minibatch:
                #get max_Q(S', a)
                old_state, action, reward, new_state = memory
                old_qval = model.predict(old_state.reshape(1,64), batch_size=1)
                newQ = model.predict(new_state.reshape(1,64), batch_size=1)
                maxQ = np.max(newQ)
                y = np.zeros((1,4))
                y[:] = old_qval[:]
                
                if reward  == -1:#non-terminal state
                    update = reward + (gamma * maxQ)
                else: #terminal state
                    update = reward
                y[0][action] = update
                X_train.append(old_state.reshape(64,))
                Y_train.append(y.reshape(4,))
            
                 
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
            #print('Game #: %s' % (i,))       
            model.fit(X_train, Y_train, batch_size=batchSize, nb_epoch=1, verbose=0)      
            state = new_state          
        if reward != -1: #if terminal state, update game status
            status = 0
        clear_output(wait=True)
    if epsilon > 0.1: #decrement over time
        epsilon -= (1/epochs)
        
        
def testAlgo(init=0):
    i = 0
    if init==0:
        state = initGrid()
    elif init==1:
        state = initGridPlayer()
    elif init==2:
        state = initGridRand()

    print("Initial State:")
    print(dispGrid(state))
    status = 1
    #while game still in progress
    while(status == 1):
        qval = model.predict(state.reshape(1,64), batch_size=1)
        action = (np.argmax(qval)) #take action with highest Q-value
        print('Move #: %s; Taking action: %s' % (i, action))
        state = makeMove(state, action)
        print(dispGrid(state))
        reward = getReward(state)
        if reward != -1:
            status = 0
            print("Reward: %s" % (reward,))
        i += 1 #If we're taking more than 10 actions, just stop, we probably can't win this game
        if (i > 10):
            print("Game lost; too many moves.")
            break
        
#testAlgo(init=1)
























