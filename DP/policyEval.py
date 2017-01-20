#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 14:37:09 2017

@author: karl
"""

import numpy as np
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv


env = GridworldEnv()

def policy_eval(policy, env, discount_factor=1.0, theta=0.00001):
    """
    Evaluate a policy given an environment and a full description of the environment's dynamics.
    
    Args:
        policy: [S, A] shaped matrix representing the policy.
        env: OpenAI env. env.P represents the transition probabilities of the environment.
            env.P[s][a] is a (prob, next_state, reward, done) tuple.
        theta: We stop evaluation once our value function change is less than theta for all states.
        discount_factor: gamma discount factor.
    
    Returns:
        Vector of length env.nS representing the value function.
    """
    # Start with a random (all 0) value function
    v = np.zeros(env.nS)
    count = 0
    minChange = 10000
    while minChange > theta:
        # TODO: Implement!
        vCurrent = v.copy()
        for i in range(env.nS):
            futureValue = 0
            for a in env.P[i]:
                nextState = env.P[i][a][0][1]
                reward = env.P[i][a][0][2]  
                futureValue += policy[i][a] * vCurrent[nextState] 
            v[i] = reward + discount_factor * futureValue
        minChange = max(abs(v-vCurrent))
        count += 1
    print(count)
    return np.array(v)



#%%
random_policy = np.ones([env.nS, env.nA]) / env.nA
v = policy_eval(random_policy, env)

#%%
# Test: Make sure the evaluated policy is what we expected
expected_v = np.array([0, -14, -20, -22, -14, -18, -20, -20, -20, -20, -18, -14, -22, -20, -14, 0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)