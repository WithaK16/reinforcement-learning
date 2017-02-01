import gym
import matplotlib
import numpy as np
import sys

from collections import defaultdict
if "../" not in sys.path:
  sys.path.append("../")
from lib.envs.blackjack import BlackjackEnv
from lib import plotting

matplotlib.style.use('ggplot')

env = BlackjackEnv()


def make_epsilon_greedy_policy(Q, epsilon, nA):
    """
    Creates an epsilon-greedy policy based on a given Q-function and epsilon.

    Args:
        Q: A dictionary that maps from state -> action-values.
            Each value is a numpy array of length nA (see below)
        epsilon: The probability to select a random action . float between 0 and 1.
        nA: Number of actions in the environment.

    Returns:
        A function that takes the observation as an argument and returns
        the probabilities for each action in the form of a numpy array of length nA.

    """

    def policy_fn(observation):
        # Implement this!
        score, dealer_score, usable_ace = observation

        actionValue = Q[(score, dealer_score, usable_ace)]
        bestAction = np.argmax(actionValue)

        """
        greedy epsilon policy value proba is epsilon/nA for every non greedy move
        1 - epsilon + (epsilon/nA) otherwise
        """
        greedyEpsilonPolicy = np.zeros(nA)
        greedyEpsilonPolicy[:] = epsilon/nA
        greedyEpsilonPolicy[bestAction] = 1 - epsilon + (epsilon/nA)
        return greedyEpsilonPolicy

    return policy_fn


def mc_control_epsilon_greedy(env, num_episodes, discount_factor=1.0, epsilon=0.1):
    """
    Monte Carlo Control using Epsilon-Greedy policies.
    Finds an optimal epsilon-greedy policy.

    Args:
        env: OpenAI gym environment.
        num_episodes: Nubmer of episodes to sample.
        discount_factor: Lambda discount factor.
        epsilon: Chance the sample a random action. Float betwen 0 and 1.

    Returns:
        A tuple (Q, policy).
        Q is a dictionary mapping state -> action values.
        policy is a function taht takes an observation as an argument and returns
        action probabilities
    """

    # Keeps track of sum and count of returns for each state
    # to calculate an average. We could use an array to save all
    # returns (like in the book) but that's memory inefficient.
    returns_sum = defaultdict(float)
    returns_count = defaultdict(float)

    # The final action-value function.
    # A nested dictionary that maps state -> (action -> action-value).
    Q = defaultdict(lambda: np.zeros(env.action_space.n))

    # The policy we're following
    policy = make_epsilon_greedy_policy(Q, epsilon, env.action_space.n)

    # Implement this!
    for i_episode in range(num_episodes):

        if i_episode % 1000 == 0:
            print("\rEpisode {}/{}.".format(i_episode, num_episodes))
            sys.stdout.flush()

        state = env.reset()
        # episodes is a list of tuple (state, action, reward)
        episodes = []
        done = False

        while not done:
            probs = policy(state)
            action = np.random.choice(np.arange(len(probs)), p=probs)
            next_state, reward, done, _ = env.step(action)
            episodes.append((state, action, reward))
            if done:
                break
            state = next_state

        listStateVisited = []
        for i, episode in enumerate(episodes):
            state = tuple(episode[0])
            action = episode[1]
            state_action_pair = (state, action)
            if state_action_pair in listStateVisited:
                pass
            else:
                listStateVisited.append(state_action_pair)
                returns_count[state_action_pair] += 1.0

                sumReturns = sum([rew[2] * (discount_factor ** j) for j, rew in enumerate(episodes[i:])])

                returns_sum[state_action_pair] += sumReturns
                Q[state][action] = returns_sum[state_action_pair] / returns_count[state_action_pair]


    return Q, policy

# Q, policy = mc_control_epsilon_greedy(env, num_episodes=1, epsilon=0.1)
Q, policy = mc_control_epsilon_greedy(env, num_episodes=500000, epsilon=0.1)


# For plotting: Create value function from action-value function
# by picking the best action at each state
V = defaultdict(float)
for state, actions in Q.items():
    action_value = np.max(actions)
    V[state] = action_value
plotting.plot_value_function(V, title="Optimal Value Function")