#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 11:41:47 2019

@author: fiestaissocrazy
"""
import pandas as pd
import numpy as np
import sys
import env_for_4states_parametric as Sim
#import display_env as Sim
import random
import pygame
import pygame.event
import pickle

pkl_file = open('Q_param_top_table.p', 'rb')
data1 = pickle.load(pkl_file)
print(data1)
params = Sim.initialise_display(1)
actions=[0,1]
episodes=10000
#def get_state(x_pos,y_pos,x_vel,y_vel,mass,moment_inertia,angular_vel,Kinetic_en,angle):
#    
#    #states=[0,1,2,3]
#    if(angle>0):
#        if(x_vel>0):
#            state=0
#        elif(x_vel<0):
#            state=1   
#    elif(angle<0):
#        if(x_vel<0):
#            state=2
#        elif(x_vel>0):
#            state=3
#    return state
def get_state(x_pos,y_pos,x_vel,y_vel,mass,moment_inertia,angular_vel,Kin_en,angle):
    #def get_state(x_pos,y_pos,x_vel,y_vel,mass,angle,Kinetic_en):#moment_inertia,angular_vel,angle,Kinetic_en):
    #states=[0,1,2,3]
    if(angle<0):
        if(x_vel<0):
            state=0
        elif(x_vel>0):
            state=1  
    elif(angle>0):
        if(x_vel>0):
            state=2
        elif(x_vel<0):
            state=3
    return state
output,params = Sim.stepper_display(*params,0)

while True:

    state = get_state(*output)
    best_action=data1[state].index(max(data1[state]))
    print(best_action,output[8], state)
#    if state == 0:
#        action = qtable['Pos(+)vel(+)'].idxmax()             
#    elif state == 1:
#        action = qtable['Pos(+)vel(-)'].idxmax()   
#    elif state == 2:
#        action = qtable['Pos(-)vel(-)'].idxmax()   
#    elif state == 3:
#        action = qtable['Pos(-)vel(+)'].idxmax()   
        
    output,params = Sim.stepper_display(*params,best_action)

    for event in pygame.event.get():
        if (event.type == pygame.QUIT):
            pygame.display.quit()
            pygame.quit()
            sys.exit(0)
        elif (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            pygame.display.quit()
            pygame.quit()
            sys.exit(0)
           # pygame.event.wait()
#    
            
#q = {
# 'Actions':[0,1], 
# 'Pos(+)vel(-)':[3,4], 
# 'Pos(+)vel(+)':[1,2],
# 'Pos(-)vel(+)':[7,8],
# 'Pos(-)vel(-)':[5,6]}
pygame.display.quit()
pygame.quit()

