# PacmanValueIterationAgent.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import util

from agents.learningAgents import ValueEstimationAgent
from game import Grid, Actions, Directions
import math
from pacman import GameState
import random
import numpy as np


class Q1Agent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        Q1 agent is a ValueIterationAgent takes a Markov decision process
        (see pacmanMDP.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp="PacmanMDP", discount=0.6, iterations=500, pretrained_values=None):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        mdp_func = util.import_by_name('./', mdp)
        self.mdp_func = mdp_func

        print('[Q1Agent] using mdp ' + mdp_func.__name__)

        self.discount = float(discount)
        self.iterations = int(iterations)

        if pretrained_values:
            self.values = np.loadtxt(pretrained_values)
        else:
            self.values = None

    ########################################################################
    ####            CODE FOR YOU TO MODIFY STARTS HERE                  ####
    ########################################################################

    def registerInitialState(self, state: GameState):

        # set up the mdp with the agent starting state
        self.MDP = self.mdp_func(state)

        # if we haven't solved the mdp yet or are not using pretrained weights
        if self.values is None:

            print("solving MDP")
            possible_states = self.MDP.getStates()
            self.values = np.zeros((self.MDP.grid_width, self.MDP.grid_height))

            # Write value iteration code here
            "*** YOUR CODE STARTS HERE ***"
            for _ in range(self.iterations):
                new_values = np.copy(self.values)
                for possible_state in possible_states:
                    max_value = -math.inf
                    possible_actions = self.MDP.getPossibleActions(possible_state)
                    for action in possible_actions:
                        transStateProbs = self.MDP.getTransitionStatesAndProbs(possible_state, action)
                        value = 0.0
                        for trans in transStateProbs:
                            value += trans[1] * (self.MDP.getReward(possible_state, action, trans[0]) +
                                                 self.discount * self.values[trans[0]])

                        max_value = max(value, max_value)

                    if new_values[possible_state[0], possible_state[1]] == -math.inf:
                        new_values[possible_state[0], possible_state[1]] = 0.0
                    else:
                        new_values[possible_state[0], possible_state[1]] = max_value

                self.values = new_values
            "*** YOUR CODE ENDS HERE ***"

            np.savetxt(f"./logs/{state.data.layout.layoutFileName[:-4]}.model", self.values,
                       header=f"{{'discount': {self.discount}, 'iterations': {self.iterations}}}")

    def computeQValueFromValues(self, state, action):
        """
        Compute the Q-value of action in state from the
        value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        trans_states_probs = self.MDP.getTransitionStatesAndProbs(state, action)
        value = 0.0

        for trans in trans_states_probs:
            value += trans[1] * (self.MDP.getReward(state, action, trans[0]) + self.discount * self.values[trans[0]])

        return value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """

        "*** YOUR CODE HERE ***"
        possible_actions = self.MDP.getPossibleActions(state)
        best_action = None
        best_action_value = -math.inf

        for action in possible_actions:
            curr_action_value = self.computeQValueFromValues(state, action)
            if curr_action_value > best_action_value:
                best_action_value = curr_action_value
                best_action = action

        return best_action

    ########################################################################
    ####            CODE FOR YOU TO MODIFY ENDS HERE                    ####
    ########################################################################

    def getValue(self, state):
        """
        Takes an (x,y) tuple and returns the value of the state (computed in __init__).
        """
        return self.values[state[0], state[1]]

    def getPolicy(self, state):
        pacman_loc = state.getPacmanPosition()
        return self.computeActionFromValues(pacman_loc)

    def getAction(self, state: GameState):
        "Returns the policy at the state "

        pacman_location = state.getPacmanPosition()
        if self.MDP.isTerminal(pacman_location):
            raise util.ReachedTerminalStateException("Reached a Terminal State")
        else:
            best_action = self.getPolicy(state)
            return self.MDP.apply_noise_to_action(pacman_location, best_action)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


