# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 18:52:29 2019

@author: verlo
"""

import env_for_4states as Sim
import random
import pygame
import pickle
import numpy as np
import pygame.event
import matplotlib.pyplot as plt
import class_q_rotational as claws
import sys
params = Sim.initialise(0)
q_table_state_index_adjustment=180
actions=[0,1]
episodes=0
saved_value=[]

def get_state(x_pos,y_pos,x_vel,y_vel,mass,angle,Kin_en):
    #states=[0,1,2,3]
    if(angle>0):
        if(x_vel>0):
            state=0
        elif(x_vel<0):
            state=1   
    elif(angle<0):
        if(x_vel<0):
            state=2
        elif(x_vel>0):
            state=3
    return state
def get_energy(x_pos,y_pos,x_vel,y_vel,mass,angle,Kin_en):
    g=9.8
    PE=mass*y_pos*g
    #KE=mass*((x_vel**2)+(y_vel**2))/2
    KE=Kin_en
    return PE+KE
#episode_number=[]
no_episodes=1001
eps=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
episode_number=np.arange(0,no_episodes,1)
cummulative_reward=0
list_data=[[[] for i in range(len(episode_number))] for e in range(len(eps))]
for j in range(1):
    for e in range(1):
        ql=claws.QLearn(actions, epsilon=0.1, alpha=0.2, gamma=0.7)
        #saved_values.append([])
        params = Sim.initialise(0)
        q_table_state_index_adjustment=180
        actions=[0,1]
        episodes=0
        episode=0
        big_file=[]
        max_angle=0
        while episode<no_episodes:
            output,params = Sim.stepper(*params,0)
            reference_y=0
            max_height = 0
        
            state=get_state(*output)
            #print(state)
            action=ql.chooseAction(state)
            previous_energy=get_energy(*output)
            center_counter=0
            max_angle=0
            M=[]
            #print(i)
            while (center_counter<10):
                
                output,params = Sim.stepper(*params,action)
                if(abs(output[5]>max_angle)):
                    max_angle=abs(output[5])
                #print(max_angle)
                next_state=get_state(*output)
                next_action=ql.chooseAction(next_state)
                if (output[1]-reference_y)>max_height:
                    max_height = output[1] - reference_y
                   # print(max_height)
                
                
                energy=get_energy(*output)
                reward=energy-previous_energy  
                cummulative_reward+=reward
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
            
            #print(big_file)
            #pickle.dump(ql.q, open("Q_table.p", "wb"),protocol=2)
            #my_q_table = myQ.Q_table
            print(eps[e])
            #.append(episode)
            saved_value.append(cummulative_reward)
            #list_data[e][episode].append(max_angle)
            episode+=1
        pygame.display.quit()
        pygame.quit()
#plt.plot(eps,episode_number)
to_plot_std=[]
for e in range(len(eps)):
    to_plot_std.append([])
    for episode in range(len(episode_number)):
        to_plot_std[e].append(np.std(np.array(list_data[e][episode])))
fig = plt.figure(1)
ax = fig.add_subplot(111)
#for i in range(len(eps)):
#    ax.plot(episode_number,saved_values[i], label='gamma='+str(eps[i]))
#handles, labels = ax.get_legend_handles_labels()
#lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.2))
#text = ax.text(-0.2,1.05, "", transform=ax.transAxes)
##text = ax.text(-0.2,1.05, "Aribitrary text", transform=ax.transAxes)
#ax.set_title("Evolution of angle reached in training\n for varying gamma values\n epsilon=0.1 alpha=0.2")
#ax.set_xlabel('Episode number')
#ax.set_ylabel('Angle (degrees)')
#ax.grid('on')
#fig.savefig(r'D:\usb\backwards_q'+r'\gamma_var.svg', format='svg',bbox_extra_artists=(lgd,text), bbox_inches='tight')
#
ax.plot(episode_number, saved_value)# label='gamma='+str(eps[i]))
#handles, labels = ax.get_legend_handles_labels()
#lgd = ax.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5,-0.2))
#text = ax.text(-0.2,1.05, "", transform=ax.transAxes)
#text = ax.text(-0.2,1.05, "Aribitrary text", transform=ax.transAxes)
ax.set_title("Cummulative reward over time for\n epsilon=0.1 alpha=0.2 gamma=0.7")
ax.set_xlabel('Episode number')
ax.set_ylabel('Cummulative reward')
ax.grid('on')
fig.savefig(r'D:\usb\backwards_q'+r'\cumm_rew.svg', format='svg')#, bbox_inches='tight')