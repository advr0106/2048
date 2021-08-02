from BaseAI import BaseAI
import time
from math import log

INF = pow(10, 9)
maxTime = 0.2
deltaTime = 0.0000000001

class PlayerAI(BaseAI):
    
    def __init__(self):
        
        super().__init__()
        self.startTime = 0
        self.depthLimit = 0
        
    def getMove(self, grid):
        
        self.startTime = time.clock()
        best_state = State(grid, -INF, 0)
        
        for lim in range(2, 50):
            #print("Depth limit at :", lim)
            self.depthLimit = lim
            init_state = State(grid, -INF, 0)
            new_state = self.maximize(init_state, -INF, INF, 0)
            if not self.checkTime():
                best_state = new_state.copy()
            else:
                break
        
        #print("\nBest Move:", best_state.getMove())
        return best_state.getMove()
    
    def checkTime(self):
        
        return time.clock() - self.startTime > maxTime - deltaTime
    
    def getAdjacentCellStats(self, grid):
        
        total_equal = 0
        total_diff = 0
        
        for i in range(grid.size):
            for j in range(grid.size):
                cell = grid.getCellValue((i, j))
                test_pos = [(i + 1, j), (i - 1, j), (i, j + 1), (i, j - 1)]
                for pos in test_pos:
                    if pos[0] >= 0 and pos[0] < grid.size and pos[1] >= 0 and pos[1] < grid.size:
                        if cell == grid.getCellValue(pos):
                            total_equal += 1
                        elif cell != 0 and grid.getCellValue(pos) != 0:
                            total_diff += abs(log(cell, 2) - log(grid.getCellValue(pos), 2))
                
        return total_equal, total_diff
    
    def getOrderedColumns(self, grid):
        
        total_asc = 0
        total_des = 0
        
        for i in range(grid.size):
            column = []
            for j in range(grid.size):
                cell = grid.getCellValue((j, i))
                column.append(cell)
            if all(column[i] <= column[i + 1] for i in range(grid.size - 1)):
                total_asc += 1
            if all(column[i] >= column[i + 1] for i in range(grid.size - 1)):
                total_des += 1

            
        return max(total_des, total_asc)
    
    def getOrderedRows(self, grid):
        
        total_asc = 0
        total_des = 0
        
        for i in range(grid.size):
            row = []
            for j in range(grid.size):
                cell = grid.getCellValue((i, j))
                row.append(cell)
            if all(row[i] <= row[i + 1] for i in range(grid.size - 1)):
                total_asc += 1
            if all(row[i] >= row[i + 1] for i in range(grid.size - 1)):
                total_des += 1
            
        return max(total_des, total_asc)
    
    def maxOnCorner(self, grid):
        
        maxValue = grid.getMaxTile()
        size = grid.size
        positions = [(0, 0), (size - 1, 0), (0, size - 1), (size - 1, size - 1)]
        for pos in positions:
            if grid.getCellValue(pos) == maxValue:
                return True
        return False
    
    def getBoardMagnitude(self, grid):
        
        total = 0
        for i in range(grid.size):
            for j in range(grid.size):
                val = grid.getCellValue((i, j))
                if val != None and val != 0:
                    total += log(val, 2)
        return total

    def getHeuristics(self, grid):
        
        """
        | Heuristics to find a closer answer to the optimal solution |
        
        DEFINITION                                      RANGE                   SCALED RANGE
        1) Number of empty cells                        [0 - 16]                [0 - 16]
        2) Number of equal value adjacent cells         [0 - 48]     * (1/3)    [0 - 16]
        3) Number of ordered rows and columns           [0 - 8]      * 2        [0 - 16]
        4) Total difference between tiled cells         [0 - 624]    * (1/39)   [0 - 16]
        5) Max value on corner                          [0 - 1]      * 16       [0 - 16]
        6) Total magnitude of cells                     [0 - 208]    * (1/13)   [0 - 16]
        """
        
        cell_stats = self.getAdjacentCellStats(grid)
        h1 = len(grid.getAvailableCells())
        h2 = cell_stats[0] * (1/3)
        h3 = (self.getOrderedRows(grid) + self.getOrderedColumns(grid)) * 2
        h4 = cell_stats[1] * (1/39)
        h5 = self.maxOnCorner(grid) * 16
        h6 = self.getBoardMagnitude(grid)* (1/13)
        
        return h1, h2, h3, h4, h5, h6
    
    def getTotalUtility(self, state):
        
        values = self.getHeuristics(state.getGrid())
        weights = [5, 0.1, 1.2, -0.1, 0.2, -0.1]
        utility = 0
        
        for i in range(len(values)):
            utility += values[i] * weights[i]
            
        return utility
    
    def terminalState(self, depth):
        
        if depth > self.depthLimit:
            return True
        
        return False
    
    def minimize(self, state, alpha, beta, depth):
        
        if self.terminalState(depth):
            
            return State(state.getGrid().clone(), state.getMove(), self.getTotalUtility(state))
        
        min_state = State(None, None, INF)
        
        for child in state.getComputerChildren():
            
            if self.checkTime():
                return min_state
            
            new_state = self.maximize(child, alpha, beta, depth + 1)
            
            if new_state.getUtil() < min_state.getUtil():
                min_state = new_state.copy()
                
            if min_state.getUtil() <= alpha:
                break
            
            if min_state.getUtil() < beta:
                alpha = min_state.getUtil()
                
        return min_state
    
    def maximize(self, state, alpha, beta, depth):
        
        if self.terminalState(depth):
            
            return State(state.getGrid().clone(), state.getMove(), self.getTotalUtility(state))
        
        max_state = State(None, None, -INF)
        
        for child in state.getPlayerChildren():
            
            if self.checkTime():
                return max_state
            
            new_state = self.minimize(child, alpha, beta, depth + 1)
                
            if new_state.getUtil() > max_state.getUtil() or max_state.getMove() == None:
                max_state = new_state.copy()
                max_state.setMove(child.getMove())
                
            if max_state.getUtil() >= beta:
                break
            
            if max_state.getUtil() > alpha:
                alpha = max_state.getUtil()
                
        return max_state
                
    
class State():
    
    def __init__(self, grid, move, util):
        
        self.grid = grid
        self.move = move
        self.util = util
        
    def getGrid(self):
        
        return self.grid
    
    def getMove(self):
        
        return self.move
    
    def getUtil(self):
        
        return self.util
    
    def setGrid(self, grid):
        
        self.grid = grid
    
    def setMove(self, move):
        
        self.move = move
        
    def setUtil(self, util):
        
        self.util = util
        
    def copy(self):
        
        new_state = State(None, None, None)
        new_state.setUtil(self.getUtil())
        new_state.setMove(self.getMove())
        
        if self.getGrid() != None:
            new_state.setGrid(self.getGrid().clone())
            
        return new_state
        
    def getPlayerChildren(self):
        
        moves = self.grid.getAvailableMoves()
        children = []
        
        for t in moves:
            copyGrid = self.getGrid().clone()
            copyGrid.move(t)
            child = State(copyGrid, t, self.getUtil())
            children.append(child)
            
        return children
            
    def getComputerChildren(self):
        
        emptyCells = self.getGrid().getAvailableCells()
        children = []
        posibleTiles = (2, 4)
        
        for cell in emptyCells:
            for tile in posibleTiles:
                copyGrid = self.getGrid().clone()
                copyGrid.setCellValue(cell, tile)
                child = State(copyGrid, self.getUtil(), None)
                children.append(child)
                
        return children