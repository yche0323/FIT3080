import logging

import util
from problems.q1a_problem import q1a_problem


def q1a_solver(problem: q1a_problem):
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    start_pos = start_state.getPacmanPosition()
    w = 10

    food = start_state.getFood()
    food_pos = (0, 0)
    for x in range(food.width):
        for y in range(food.height):
            if food.data[x][y] is True:
                food_pos = (x, y)

    start_to_food = util.manhattanDistance(start_pos, food_pos)
    open_list = [[start_state, start_pos, start_to_food, [], 0]]
    closed_list = []

    while len(open_list) > 0:
        expanding = open_list.pop(0)
        closed_list.append(expanding)
        successors = problem.getSuccessors(expanding[0])
        succ_states = successors[0]
        succ_pac_pos = successors[1]
        succ_actions = successors[2]
        cost = successors[3]

        for s in range(len(succ_states)):
            new_state = succ_states[s]
            new_pos = succ_pac_pos[s]
            new_action = succ_actions[s]
            if problem.isGoalState(new_state.getNumFood()):
                return expanding[3] + [new_action]
            else:
                found = [False, 0]
                for i in range(len(closed_list)):
                    if new_pos == closed_list[i][1]:
                        found = [True, i]
                        break

                if not found[0]:
                    hn = util.manhattanDistance(new_pos, food_pos)
                    gn = expanding[4] + cost
                    new_actions = expanding[3] + [new_action]
                    open_list.append([new_state, new_pos, hn, new_actions, gn])
                else:
                    gn = expanding[4] + cost
                    if gn < closed_list[found[1]][4]:
                        closed_list[found[1]][4] = gn
                        open_list.append(closed_list[found[1]])
                        closed_list.pop(found[1])

        open_list = sorted(open_list, key=lambda x: w * x[2] + x[4])
