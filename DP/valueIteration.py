#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 12 16:31:28 2017

@author: karl
"""

import numpy as np
import pprint
import sys
if "../" not in sys.path:
  sys.path.append("../") 
from lib.envs.gridworld import GridworldEnv
from policyIteration import policy_eval

pp = pprint.PrettyPrinter(indent=2)
env = GridworldEnv()


def value_iteration(env, theta=0.0001, discount_factor=1.0):
    """
    Value Iteration Algorithm.
    
    Args:
        env: OpenAI environment. env.P represents the transition probabilities of the environment.
        theta: Stopping threshold. If the value of all states changes less than theta
            in one iteration we are done.
        discount_factor: lambda time discount factor.
        
    Returns:
        A tuple (policy, V) of the optimal policy and the optimal value function.        
    """
        
    def one_look_ahead(s, v):
            """
            Helper function to calculate the value for all action in a given state.
            
            Args:
                state: The state to consider (int)
                V: The value to use as an estimator, Vector of length env.nS
            
            Returns:
                A vector of length env.nA containing the expected value of each action.
            """
            actionValue = np.zeros(env.nA)
            for a in range(env.nA):
                for prob, nextState, reward, done in env.P[s][a]:
                    actionValue[a] += prob * (reward + discount_factor * v[nextState])
            return actionValue
    v = np.zeros(env.nS)
   
    while True:
        delta = 0
        for s in range(env.nS):
            
            # Do a one-step lookahead to find the best action
            actionValue = one_look_ahead(s, v)
            bestActionValue = np.max(actionValue)
            # Calculate delta across all states seen so far
            delta = max(delta, np.abs(bestActionValue - v[s]))
            # Update the value function
            v[s] = bestActionValue  
        # Check if we can stop 
        if delta < theta:
            break    
    # Create a deterministic policy using the optimal value function
    policy = np.zeros([env.nS, env.nA])
    for s in range(env.nS):
        # One step lookahead to find the best action for this state
        A = one_look_ahead(s, v)
        best_action = np.argmax(A)
        # Always take the best action
        policy[s, best_action] = 1.0
    # Implement!
    return policy, v
    

    
    
    
    
#%%
policy, v = value_iteration(env)

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