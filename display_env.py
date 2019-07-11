"""
Created on Tue Feb 26 14:30:47 2019

@author: Ben (edited by Meghan)

Purpose: Pymunk for visualising swing



"""


import sys, random
import pygame
import pymunk
import pymunk.pygame_util
from pygame.locals import *
import numpy as np
from pygame.color import *

def add_swing(space):
    rotation_center_body = pymunk.Body(body_type = pymunk.Body.STATIC)
    rotation_center_body.position = (300,300)

    swing = pymunk.Body()
    swing.position = (300,120)
    rod = pymunk.Segment(swing, (0, 180), (0, 0), 2.0)
    rod.mass = 2.5
    seat = pymunk.Poly(swing, [(-8.5, 5), (8.5, 5), (-8.5,0),(8.5,0)])
    seat.mass = 1.065

    ceiling_joint = pymunk.PivotJoint(swing, rotation_center_body, (0,179), (0,0))
    space.add(rod, seat, swing, ceiling_joint)
    
    return swing

def add_robot(space,seat):
    #define robot upper body and upper leg
    robot_body = pymunk.Body()
    robot_body.position = seat.position + (0,5)
    robot_u = pymunk.Poly(robot_body,[(-4,31.64),(-4,0),(4,0),(4,31.64)])
    robot_u.mass = 3.311
    robot_u.color = THECOLORS["red"]
    robot_u.filter = pymunk.ShapeFilter(mask=pymunk.ShapeFilter.ALL_MASKS ^ 1)
    
    
    robot_u_leg = pymunk.Poly(seat,[(-0.5,5),(-10.5,5),(-10.5,10),(-0.5,10)])
    robot_u_leg.color = THECOLORS["red"]
    robot_u_leg.mass = 0.603
    
    #define robot lower leg
    robot_leg = pymunk.Body()
    robot_leg.position = seat.position
    robot_l_leg = pymunk.Poly(robot_leg,[(-10.5,5),(-10.5,-11.8),(-4.0,-11.8),(-4.0,5)])
    robot_l_leg.mass = 1.214
    robot_l_leg.color = THECOLORS["red"]
    space.add(robot_body,robot_u,robot_u_leg,robot_leg,robot_l_leg)
    
    #motor and pivot for hip
    seat_motor = pymunk.SimpleMotor(seat,robot_body,0)
    seat_motor.max_force = 1e6
    seat_pivot = pymunk.PivotJoint(seat,robot_body,robot_body.position)
    seat_pivot._set_collide_bodies(False)
    seat_pivot_lim = pymunk.RotaryLimitJoint(robot_body,seat,0,1.11529)
    space.add(seat_motor,seat_pivot,seat_pivot_lim)
    
    #motor and pivot for knee
    knee_motor = pymunk.SimpleMotor(seat,robot_leg,0)
    knee_motor.max_force = 1e5
    knee_pivot = pymunk.PivotJoint(seat,robot_leg,seat.position+(-8,7))
    knee_pivot._set_collide_bodies(False)
    knee_pivot_lim = pymunk.RotaryLimitJoint(seat,robot_leg,-1.04604,0.44604)
    space.add(knee_motor,knee_pivot,knee_pivot_lim)

    return seat_motor,knee_motor,robot_body

def initialise():
    damping_coeff = np.exp(-2.56*np.deg2rad(0.0034))

    space = pymunk.Space()
    space.gravity = (0, -981)
    space.damping = damping_coeff

    swing = add_swing(space)
    seat_motor,knee_motor,robot_u = add_robot(space,swing)
    swing.apply_force_at_local_point((-40000,0),(0,0))

    pygame.init()
    screen = pygame.display.set_mode((600, 600))
    pygame.display.set_caption("I'm so swungover")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 16)
    draw_options = pymunk.pygame_util.DrawOptions(screen)
    return [space,swing,robot_u,seat_motor,knee_motor,screen,draw_options,clock,font]

def stepper(space,swing,robot_u,seat_motor,knee_motor,screen,draw_options,clock,font,action):
    if action == 2:
        seat_motor.rate = 2.788225
        knee_motor.rate = 3.7302
    elif action == 1:
        seat_motor.rate = 0
        knee_motor.rate = 0
    elif action == 0:
        seat_motor.rate = -2.788225
        knee_motor.rate = -3.7302
        
    screen.fill((255,255,255))
    space.debug_draw(draw_options)
    screen.blit(font.render("Angle = " + str(np.rad2deg(swing.angle)),1, THECOLORS["black"]), (0,0))
    screen.blit(font.render("Swing Velocity = " + str(swing._get_angular_velocity()),1, THECOLORS["black"]), (0,20))
    pygame.display.flip()
    clock.tick(50)
    
    for _ in range(5):
        space.step(1.0/100.0)
    
    #return values
    angle = np.rad2deg(swing.angle)
    vel = [swing.velocity[0],swing.velocity[1]]
    body_pos = robot_u.angle
    
    return [angle, vel, body_pos],[space,swing,robot_u,seat_motor,knee_motor,screen,draw_options,clock,font]
    
