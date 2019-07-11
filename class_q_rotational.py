# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 18:12:30 2019

@author: verlo
"""
 
import random
import math
def linear_epsilon(old_epsilon):
     new_epsilon=old_epsilon-0.1
     return new_epsilon
def exp_epsilon(old_epsilon):
    new_epsilon=old_epsilon/math.exp(0.004)
    return new_epsilon

class QLearn:
    def __init__(self, actions, epsilon=0.1, alpha=0.2, gamma=0.9):
        #print(len(actions))
        self.q = [[0,0] for i in range(4) ] 

        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma
        self.actions = actions

    def getQ(self, state, action):
        
        return self.q[state][action]
        # return self.q.get((state, action), 1.0)

    def learnQ(self, state, action, reward, value):
        oldv = self.q[state][action]
        if oldv is 0:
            self.q[state][action] = reward
        else:
            self.q[state][action] = oldv + self.alpha * (value - oldv)

    def chooseAction(self, state):
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q = [self.getQ(state, a) for a in self.actions]
            maxQ = max(q)
            count = q.count(maxQ)
            if count > 1:
                best = [i for i in range(len(self.actions)) if q[i] == maxQ]
                i = random.choice(best)
            else:
                i = q.index(maxQ)

            action = self.actions[i]
        return action

    def learn_Ql(self, state1, action1, reward, state2):
        maxqnew = max([self.getQ(state2, a) for a in self.actions])
        self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)

    def learn_SARSA(self, state1, action1, reward, state2, action2):
        qnext = self.getQ(state2, action2)
        self.learnQ(state1, action1, reward, reward + self.gamma * qnext)
    def printQ(self):
        keys = self.q.keys()
        states = list(set([a for a,b in keys]))
        actions = list(set([b for a,b in keys]))
        
        dstates = ["".join([str(int(t)) for t in list(tup)]) for tup in states]
        print (" "*4) + " ".join(["%8s" % ("("+s+")") for s in dstates])
        for a in actions:
            print ("%3d " % (a)) + \
                " ".join(["%8.2f" % (self.getQ(s,a)) for s in states])

    def printV(self):
        keys = self.q.keys()
        states = [a for a,b in keys]
        statesX = list(set([x for x,y in states]))
        statesY = list(set([y for x,y in states]))

        print (" "*4) + " ".join(["%4d" % (s) for s in statesX])
        for y in statesY:
            maxQ = [max([self.getQ((x,y),a) for a in self.actions])
                    for x in statesX]
            print ("%3d " % (y)) + " ".join([ff(q,4) for q in maxQ])
        
import math
def ff(f,n):
    fs = "{:f}".format(f)
    if len(fs) < n:
        return ("{:"+n+"s}").format(fs)
    else:
        return fs[:n]
    # s = -1 if f < 0 else 1
    # ss = "-" if s < 0 else ""
    # b = math.floor(math.log10(s*f)) + 1
    # if b >= n:
    #     return ("{:" + n + "d}").format(math.round(f))
    # elif b <= 0:
    #     return (ss + ".{:" + (n-1) + "d}").format(math.round(f * 10**(n-1)))
    # else:
    #     return ("{:"+b+"d}.{:"+(n-b-1)+"

