import logging
import math
import random

import util
from game import Actions, Agent, Directions
from logs.search_logger import log_function
from pacman import GameState
from util import manhattanDistance


def scoreEvaluationFunction(currentGameState: GameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    pacman_pos = currentGameState.getPacmanPosition()
    food = currentGameState.getFood()
    food_pos = []
    for x in range(food.width):
        for y in range(food.height):
            if food.data[x][y] is True:
                food_pos.append((x, y))

    if len(food_pos) <= 0:
        nearest_food = 0
    else:
        nearest_food = min([manhattanDistance(pacman_pos, food_pos[i]) for i in range(len(food_pos))])
        nearest_food = 1 / (nearest_food + 1)  # The closer the food, the higher the score

    return nearest_food + currentGameState.getScore()


class Q2A_Agent(Agent):

    def __init__(self, evalFn='scoreEvaluationFunction', depth='3'):
        self.index = 0  # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

    @log_function
    def getAction(self, gameState: GameState):
        """
            Returns the minimax action from the current gameState using self.depth
            and self.evaluationFunction.

            Here are some method calls that might be useful when implementing minimax.

            gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

            gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

            gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        logger = logging.getLogger('root')
        logger.info('MinimaxAgent')
        "*** YOUR CODE HERE ***"
        def alpha_beta(state: GameState, depth: int, alpha: float, beta: float, agent_index: int):
            # If search depth reached or is a leaf node, then return the evaluation value
            if depth == 0 or state.isWin() or state.isLose():
                return self.evaluationFunction(state), None

            # If the current agent is a pacman, then look for the max value (the best move for pacman)
            if agent_index == self.index:
                max_eval = -math.inf
                legal_actions = state.getLegalPacmanActions()
                if Directions.STOP in legal_actions:
                    legal_actions.remove(Directions.STOP)
                best_action = None
                for a in legal_actions:
                    succ_state = state.generatePacmanSuccessor(a)
                    succ_eval, _ = alpha_beta(succ_state, depth - 1, alpha, beta, agent_index + 1)
                    if succ_eval > max_eval:
                        max_eval = succ_eval
                        best_action = a
                    alpha = max(alpha, max_eval)
                    # Beta-cutoff
                    if beta <= alpha:
                        break
                return max_eval, best_action
            # Otherwise (i.e. the current agent is a ghost), we look for the min value (the best move for ghost)
            else:
                min_eval = math.inf
                legal_actions = state.getLegalActions(agent_index)
                next_agent_index = agent_index + 1
                if next_agent_index == state.getNumAgents():
                    next_agent_index = 0

                for a in legal_actions:
                    succ_state = state.generateSuccessor(agent_index, a)
                    succ_eval, _ = alpha_beta(succ_state, depth, alpha, beta, next_agent_index)
                    min_eval = min(min_eval, succ_eval)
                    beta = min(beta, succ_eval)
                    # Alpha-cutoff
                    if beta <= alpha:
                        break
                return min_eval, None

        _, action = alpha_beta(gameState, self.depth, -math.inf, math.inf, 0)
        return action
