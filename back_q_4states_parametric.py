# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:17:45 2019

@author: verlo
"""
import env_for_4states_parametric as Sim
import random
import pygame
import pickle
import pygame.event
import class_q as claws
import sys
params = Sim.initialise(1)
#output,params = Sim.stepper(*params,1)
q_table_state_index_adjustment=180
actions=[0,1]
episodes=10000

ql=claws.QLearn(actions, epsilon=0.1, alpha=0.2, gamma=0.9)
def get_state(x_pos,y_pos,x_vel,y_vel,mass,moment_inertia,angular_vel,Kin_en,angle):
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
def get_energy(x_pos,y_pos,x_vel,y_vel,mass,moment_inertia,angular_vel,Kin_en,angle):
    g=9.8
    PE=mass*y_pos*g
    #KE=Kin_en
    KE=(moment_inertia*angular_vel**2)/2
    #KE=mass*((x_vel**2)+(y_vel**2))/2
    return PE+KE
big_file=[]
output,params = Sim.stepper(*params,0)
for i in range(episodes):

    reference_y=0
    max_height = 0
    state=get_state(*output)
    #print(state)
    action=ql.chooseAction(state)
    previous_energy=get_energy(*output)
    center_counter=0
    M=[]
    print(i)
    while (center_counter<10):
        
        output,params = Sim.stepper(*params,action)
        next_state=get_state(*output)
        next_action=ql.chooseAction(next_state)
        if (output[1]-reference_y)>max_height:
            max_height = output[1] - reference_y
            print(max_height)
        
        
        energy=get_energy(*output)
        reward=energy-previous_energy        
        big_file.append([output,next_action,next_state,reward])
        ql.learn_SARSA(state, action, reward, next_state, next_action)
        M.append([state,action,reward,next_state])
        if(state!=next_state):
            center_counter+=1
        state=next_state
        action=next_action
        previous_energy=energy
#        for event in pygame.event.get():
#            if (event.type == pygame.QUIT):
#                pygame.display.quit()
#                pygame.quit()
#                sys.exit(0)
#            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
#                pygame.display.quit()
#                pygame.quit()
#                sys.exit(0)
        #pygame.event.wait()
    for j in range(len(M)-1,0):
        ql.learn_Ql(M[j][0], M[j][1], M[j][2], M[j][3])

print(big_file)    
pickle.dump(ql.q, open("Q_param_top_table.p", "wb" ))
#my_q_table = myQ.Q_table
pygame.display.quit()
pygame.quit()
