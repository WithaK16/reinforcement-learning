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

def policy_improvement(env, policy_eval_fn=policy_eval, discount_factor=1.0):
    """
    Policy Improvement Algorithm. Iteratively evaluates and improves a policy
    until an optimal policy is found.
    
    Args:
        env: The OpenAI envrionment.
        policy_eval_fn: Policy Evaluation function that takes 3 arguments:
            policy, env, discount_factor.
        discount_factor: Lambda discount factor.
        
    Returns:
        A tuple (policy, V). 
        policy is the optimal policy, a matrix of shape [S, A] where each state s
        contains a valid probability distribution over actions.
        V is the value function for the optimal policy.
        
    """
    # Start with a random policy
    policy = np.ones([env.nS, env.nA]) / env.nA
    while True:
        #evaluate policy
        v = policy_eval_fn(policy, env, discount_factor)
        
        isGoodPolicy = True #set to false when not a good policy
        print(v)
        for i in range(env.nS):
            chosenAction = np.argmax(policy[i])
            actionValue = np.zeros(env.nA)
            for a in env.P[i]:
                nextState = env.P[i][a][0][1]
                reward = env.P[i][a][0][2]  
                futureValue = reward + discount_factor * v[nextState]
                actionValue[a] = futureValue
            bestAction = np.argmax(actionValue)
            
            if chosenAction != bestAction:
                isGoodPolicy = False
            policy[i] = 0
            policy[i][bestAction] = 1
        
        if isGoodPolicy:
            return policy, v

#%%
policy, v = policy_improvement(env)
print("Policy Probability Distribution:")
print(policy)
print("")

print("Reshaped Grid Policy (0=up, 1=right, 2=down, 3=left):")
print(np.reshape(np.argmax(policy, axis=1), env.shape))
print("")

print("Value Function:")
print(v)
print("")

print("Reshaped Grid Value Function:")
print(v.reshape(env.shape))
print("")
#%%
# Test the value function
expected_v = np.array([ 0, -1, -2, -3, -1, -2, -3, -2, -2, -3, -2, -1, -3, -2, -1,  0])
np.testing.assert_array_almost_equal(v, expected_v, decimal=2)