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

    #Encontrar la utilidad mas grande para cuando la computadura juega 
    def Maximize(self, alpha, beta, depth, start): #self nos referimos a una instancia de la clase,
        #Alpha nos referimos a nodo  alfa, beta nodo beta, con depth nos referimos a la profundidad y start para el tiempo
        if PlayerAI.terminal(self) or depth == 0 or (time.perf_counter() - start) > 0.02: #Aqui evaluamos si la terminal esta siendo
            #ejecuta o la profundidad se compara a 0 (es decir que no tiene donde mas explorar)
            #o si el tiempo se pasa entonces retornada la instancia.eval
            return PlayerAI.Eval(self)

        maxUtility = -np.inf #Estamos asignandole el valor de -infinito para la comparacion

        #Los nodos hijos para la computadora cuando juega el juego son los cuadritos vecinos
        for child in  PlayerAI.children(self): #para los hijos del nodo 
            maxUtility = max(maxUtility,
                             PlayerAI.Minimize(self=child, alpha=alpha, beta=beta, depth=depth - 1, start=start))
            #vamos a evaluzar el maximo entre - infinito y el valor que nos devolvera el nodo minimizado y luego comparemos
            #si lo que da es mas que beta rompemos el ciclo
            if maxUtility >= beta:
                break
            #y como este cumple ese parametro le asignamos a alpha el mayor entre maxutility y alpha 
            alpha = max(maxUtility, alpha)
            
        return maxUtility

    # Encuentra la utilidad mas pequeña para a computadora cuando coloca las fichas en los cuadritos aleatoriamente. 
    def Minimize(self, alpha, beta, depth, start): #self nos referimos a una instancia de la clase,
        #Alpha nos referimos a nodo  alfa, beta nodo beta, con depth nos referimos a la profundidad y start para el tiempo
        #Aqui evaluamos si la terminal esta siendo ejecuta o la profundidad se compara a 0 (es decir que no tiene donde mas explorar)
        #o si el tiempo se pasa entonces retornada la instancia.eval que es para que se evalue mediante la heuristica
        if PlayerAI.terminal(self) or depth == 0 or (time.perf_counter() - start) > 0.02: 
            return PlayerAI.Eval(self) 

        minUtility = np.inf #Estamos asignandole el valor de infinito para la comparacion

        empty = self.getAvailableCells(); #Aqui obtenemos los cuadritos vacios

        children = [] #Aqui creamos una lista de hijos

        # Atendiendo a las casillas que estan vacias, creamos una lista de hijos
        for pos in empty:
            
            # Creamos dos copias del tablero como se encuentra actualmente. 
            current_grid2 = self.clone()
            current_grid4 = self.clone()

            # Agregamos una ficha de 2 en la posicion que nosotros le asignamos
            current_grid2.insertTile(pos, 2)
            # Agregamos una ficha de 4 en la posicion que nosotros le asignamos
            current_grid4.insertTile(pos, 4)

            # Luego agregamos estas posibilidades a los hijos
            children.append(current_grid2)
            children.append(current_grid4)

        #Los nodos hijos para cuando la computadora juega aleatorio incluyendo todas las casillas posibles para su estado actual 
        for child in children:
            #vamos a evaluzar el minimo entre infinito y el valor que nos devolvera el nodo maximizado y luego comparemos
            minUtility = min(minUtility, PlayerAI.Maximize(self=child, alpha=alpha, beta=beta, depth=depth - 1, start=start))
            #si lo que da es menor o igual que alpha rompemos el ciclo
            if minUtility <= alpha:
                break
            #y como este cumple ese parametro le asignamos a beta el menor entre minutility y beta
            beta = min(minUtility, beta)
        
        return minUtility
