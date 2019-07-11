import nodisplay_env as Sim
import random
import pygame
import pickle
import pygame.event
import class_q as claws
params = Sim.initialise_display(0)
q_table_state_index_adjustment=180
actions=[0,1,2]
episodes=10000

ql=claws.QLearn(actions, epsilon=0.1, alpha=0.2, gamma=0.9)
for i in range(episodes):
    output,params = Sim.stepper(*params,0)
    angle=int(round(output[0]))
    state=angle+q_table_state_index_adjustment
    #print(state)
    action=ql.chooseAction(state)
    center_counter=0
    previous_angle=0
    M=[]
    print(i)
    while (center_counter<10):
        
        output,params = Sim.stepper_display(*params,action)
        
        
        #if(abs(velocity)>0):
         #   reward=-abs(velocity)
        #else:
         #   reward=previous_angle**2
        
        next_angle=int(round(output[0]))
        reward=next_angle-previous_angle
            
        print(next_angle)
        next_state=next_angle+q_table_state_index_adjustment
        next_action=ql.chooseAction(next_state)        
        ql.learn_SARSA(state, action, reward, next_state, next_action)
        M.append([state,action,reward,next_state])
        state=next_state
        action=next_action
        previous_angle=next_angle
#        for event in pygame.event.get():
#            if (event.type == pygame.QUIT):
#                running = False
#            elif (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
#                running = False
#        #pygame.event.wait()
    for j in range(len(M)-1,0):
        ql.learn_Ql(M[j][0], M[j][1], M[j][2], M[j][3])
        
pickle.dump(ql.q, open("Q_table.p", "wb" ))
#my_q_table = myQ.Q_table
pygame.display.quit()
pygame.quit()
