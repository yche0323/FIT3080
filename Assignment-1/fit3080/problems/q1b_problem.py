import logging
import time
from typing import Tuple

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState


class q1b_problem:
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function
    """
    def __str__(self):
        return str(self.__class__.__module__)

    def __init__(self, gameState: GameState):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.startingGameState: GameState = gameState

    @log_function
    def getStartState(self):
        "*** YOUR CODE HERE ***"
        return self.startingGameState

    @log_function
    def isGoalState(self, state):
        "*** YOUR CODE HERE ***"
        return state == 0

    @log_function
    def getSuccessors(self, state: GameState):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """
        "*** YOUR CODE HERE ***"
        states = []
        pac_pos = []
        actions = state.getLegalPacmanActions()

        for a in actions:
            successor_state = state.generatePacmanSuccessor(a)
            states.append(successor_state)
            pac_pos.append(successor_state.getPacmanPosition())

        return states, pac_pos, actions, 1
