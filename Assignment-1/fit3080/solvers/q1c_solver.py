import logging

import util
from problems.q1c_problem import q1c_problem


def q1c_solver(problem: q1c_problem):
    "*** YOUR CODE HERE ***"
    start_state = problem.getStartState()
    start_pos = start_state.getPacmanPosition()

    food = start_state.getFood()
    food_pos = []
    for x in range(food.width):
        for y in range(food.height):
            if food.data[x][y] is True:
                food_pos.append((x, y))
    food_pos = sorted(food_pos, key=lambda x: util.manhattanDistance(start_pos, x))

    open_list = [[start_state, start_pos, [food_pos[0], util.manhattanDistance(start_pos, food_pos[0])], [], 0]]
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

            found = [False, 0]
            for i in range(len(closed_list)):
                if new_pos == closed_list[i][1]:
                    found = [True, i]
                    break

            if not found[0]:
                new_actions = expanding[3] + [new_action]

                min_hn = [food_pos[0], util.manhattanDistance(new_pos, food_pos[0])]
                for i in range(1, len(food_pos)):
                    hn = util.manhattanDistance(new_pos, food_pos[i])
                    if hn < min_hn[1]:
                        min_hn = [food_pos[i], hn]

                open_list.append([new_state, new_pos, min_hn, new_actions, expanding[4] + cost])
                closed_list.append([new_state, new_pos, min_hn, new_actions, expanding[4] + cost])

                if new_pos in food_pos:
                    food_pos.remove(new_pos)
                    optimal_path = new_actions
                    open_list = [[new_state, new_pos, min_hn, new_actions, expanding[4] + cost]]
                    closed_list = []

                    if problem.isGoalState(len(food_pos)):
                        return optimal_path

                    break
            else:
                gn = expanding[4] + cost
                if gn < closed_list[found[1]][4]:
                    closed_list[found[1]][4] = gn
                    open_list.append(closed_list[found[1]])
                    closed_list.pop(found[1])

        open_list = sorted(open_list, key=lambda x: x[2][1] + x[4])
