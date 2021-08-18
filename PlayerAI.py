from BaseAI import BaseAI
import time
from math import log
import numpy as np

class PlayerAI(BaseAI):

    def getMove(self, grid):
        moves = grid.getAvailableMoves()
        maxUtility = -np.inf
        nextDir = -1

        for move in moves:
            child = PlayerAI.getChild(grid, move)

            utility = PlayerAI.Decision(self=child, max=False)

            if utility >= maxUtility:
                maxUtility = utility
                nextDir = move

        return nextDir

    def getChild(self, dir):
        temp = self.clone()
        temp.move(dir)
        return temp

    # Gets all the Children of a node
    def children(self):
        children = []
        for move in self.getAvailableMoves():
            children.append(PlayerAI.getChild(self, move))
        return children

    # Returns true if the node is terminal
    def terminal(self):
        return not self.canMove()

    # Evaluates the heuristic. The heuristic used here is a gradient function
    def Eval(self):
        import math
        import numpy as np

        if PlayerAI.terminal(self):
            return -np.inf

        gradients = [
            [[3, 2, 1, 0], [2, 1, 0, -1], [1, 0, -1, -2], [0, -1, -2, -3]],
            [[0, 1, 2, 3], [-1, 0, 1, 2], [-2, -1, 0, 1], [-3, -2, -1, -0]],
            [[0, -1, -2, -3], [1, 0, -1, -2], [2, 1, 0, -1], [3, 2, 1, 0]],
            [[-3, -2, -1, 0], [-2, -1, 0, 1], [-1, 0, 1, 2], [0, 1, 2, 3]]
        ]

        values = [0, 0, 0, 0]

        for i in range(4):
            for x in range(4):
                for y in range(4):
                    values[i] += gradients[i][x][y] * self.map[x][y]

        return max(values)

    def Decision(self, max=True):
        limit = 4
        start = time.perf_counter()

        if max:
            return PlayerAI.Maximize(self=self, alpha=-np.inf, beta=np.inf, depth=limit, start=start)
        else:
            return PlayerAI.Minimize(self=self, alpha=-np.inf, beta=np.inf, depth=limit, start=start)

    # Finds the largest utility for the Max Player(Computer playing the game)
    def Maximize(self, alpha, beta, depth, start):
        if PlayerAI.terminal(self) or depth == 0 or (time.perf_counter() - start) > 0.02:
            return PlayerAI.Eval(self)

        maxUtility = -np.inf

        # The children for the Max player are the neighboring tiles
        for child in PlayerAI.children(self):
            maxUtility = max(maxUtility,
                             PlayerAI.Minimize(self=child, alpha=alpha, beta=beta, depth=depth - 1, start=start))

            if maxUtility >= beta:
                break

            alpha = max(maxUtility, alpha)

        return maxUtility

    # Finds the smallest utility for the Min Player(Computer placing the random tiles)
    def Minimize(self, alpha, beta, depth, start):
        if PlayerAI.terminal(self) or depth == 0 or (time.perf_counter() - start) > 0.02:
            return PlayerAI.Eval(self)

        minUtility = np.inf

        empty = self.getAvailableCells();

        children = []

        for pos in empty:
            current_grid2 = self.clone()
            current_grid4 = self.clone()

            current_grid2.insertTile(pos, 2)
            current_grid4.insertTile(pos, 4)

            children.append(current_grid2)
            children.append(current_grid4)

        # The children for the Min player include all random tile possibilities for the current state
        for child in children:
            minUtility = min(minUtility,
                             PlayerAI.Maximize(self=child, alpha=alpha, beta=beta, depth=depth - 1, start=start))

            if minUtility <= alpha:
                break

            beta = min(minUtility, beta)

        return minUtility