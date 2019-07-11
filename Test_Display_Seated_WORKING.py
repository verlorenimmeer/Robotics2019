#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:07:13 2019

****TEST DISPLAY****

@author: Meghan
"""


import numpy as np
import time
import pickle
import pygame.event
import display_env as Sim
import sys
from pygame.locals import *
import matplotlib.pyplot as plt
import class_q as q

alpha = 0.5
epsilon = 0.8
gamma = 0.9
max_swing = 90
max_reward = 1000
i_reward = 200
action_space = 2 

"VARIABLE"
graph_range = 1000


#set up class, open pickled Q table from training and set this as class variable
myQ = q.Q_Learn(max_reward, max_swing, i_reward, alpha, gamma)
myQ.Q_table = pickle.load(open("Q_table.p", "rb" ))



amp = []
x = np.arange(0,graph_range,4)


total_epochs = 0
next_action = 0
params = Sim.initialise()
#statenum = 1800
#number of episodes = 100
for i in range (0,graph_range):
    #episode resets, reward resets, done verification resets, and timestep for episode resets.
        output,params = Sim.stepper(*params,next_action)
        statenum = myQ.Get_statenum(np.around(output[0]), myQ.Get_vel_mag(output[1][0]))
        next_action = myQ.Get_max_Q(statenum)[1]
        
        
        
        
        #ensuring project flows continously 
        for event in pygame.event.get(): #ensures the program doesn't crash when you close the pygame window
            if event.type == pygame.QUIT:
                pygame.display.quit()
                pygame.quit()
                sys.exit(0)
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                pygame.display.quit()
                pygame.quit()
                sys.exit(0)
                
    #initialises first place and saves previous statenumber before the learning commences


print("Max swing angle: ", max(amp))
