# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 00:15:17 2019

@author: verlo
"""

import Param_swing as Sim
import random
import pygame
import pygame.event

params = Sim.initialise()
for _ in range(10000):
    action = random.randint(0,3)
    output,params = Sim.stepper(*params,action)
    print(output[0])
    pygame.event.wait()
pygame.quit()