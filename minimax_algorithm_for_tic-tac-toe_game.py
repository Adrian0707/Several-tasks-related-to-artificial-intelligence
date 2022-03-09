import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import networkx as nx
import pandas as pd
from networkx.drawing.nx_pydot import graphviz_layout
import random

# **************************************************************
# * Wykonał : Adrian Pruszyński                                *
# * EITI, Informatyka, 2021L                                   *
# **************************************************************


# Domyślna klasa na bazie której definiowane moga być drzewa gier
class Node:
    def __init__(self, successors, name):
        self.successors = successors
        self.name = name

# Funkcja oceny stanu dla zadanej gry
    def h(self):
        raise Exception('h should be implemented')

# Funkcja odpowiadająca za ustalenie czy wierzchołek jest terminalny
    def terminal(self):
        raise Exception('terminal should be implemented')

# Funkcja odpowiadająca za dodawanie następników
    def add_successor(self, successor):
        self.successors.append(successor)

# Funkcja odpowiadająca za wyświetlenie grafu
    def print_graph(self, chosen_nodes=[], turn=True):
            connection_from = []
            connections_to = []
            self.connections(connection_from, connections_to)
            df = pd.DataFrame({'from': connection_from, 'to': connections_to})

            G = nx.from_pandas_edgelist(df, 'from', 'to')
            pos = graphviz_layout(G, prog="dot")

            if(len(chosen_nodes) > 0):
                p = nx.shortest_path(G, self.name, chosen_nodes[len(chosen_nodes)-1])

                for e in G.edges():
                    G[e[0]][e[1]]['color'] = 'black'

                for i in range(len(p) - 1):
                    if(turn):
                        G[p[i]][p[i + 1]]['color'] = 'green' if (-1) ** i > 0 else 'red'
                    else:
                        G[p[i]][p[i + 1]]['color'] = 'green' if (-1) ** i < 0 else 'red'
                edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]

                nx.draw(G, edge_color=edge_color_list, font_size=8, with_labels=True, pos=pos)
                red_patch = mpatches.Patch(color='red', alpha=0.5, label='MIN')
                green_patch = mpatches.Patch(color='green', alpha=0.5, label='MAX')
                plt.legend(handles=[red_patch, green_patch])
            else:
                nx.draw(G, with_labels=True, pos=pos)

            plt.show()

# Funkcja odpowiadająca za utwożenie listy połączeń w grafie
    def connections(self, connection_from, connections_to):
            for successor in self.successors:
                connection_from.append(self.name)
                connections_to.append(successor.name)
                successor.connections(connection_from, connections_to)


# Klasa służąca do zdefiniowania drzewa gry kółko i krzyżyk
class NodeTicTacToe(Node):

    def __init__(self, successors, board, name):
        self.board = board
        self.heuristics_board_wages = [[3, 2, 3], [2, 4, 2], [3, 2, 3]]
        np_arr = np.array(board)
        np_arrN = np.where(np_arr == 1, 'O', np_arr)
        np_arrN = np.where(np_arr == 0, ' ', np_arrN)
        np_arrN = np.where(np_arr == -1, 'X', np_arrN)
        super().__init__(successors,
                         f"h(s)={self.h()}\n{np.array2string(np_arrN, precision=2, separator='|', suppress_small=True)}\nid={name}\n")

    def h(self):
        sum = 0
        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                sum += self.board[i][j] * self.heuristics_board_wages[i][j]
        return sum

    def terminal(self):
        if(len(self.successors) == 0):
            return True
        if((math.fabs(sum(self.board[0])) == 3)
                | (math.fabs(sum(self.board[1])) == 3)
                | (math.fabs(sum(self.board[2])) == 3)
                | (math.fabs(self.board[0][0] + self.board[1][1] + self.board[2][2]) == 3)
                | (math.fabs(self.board[0][2] + self.board[1][1] + self.board[2][0]) == 3)):
            return True
        for i in range(0, 3, 1):
            if(math.fabs(self.board[0][i] + self.board[1][i] + self.board[2][i]) == 3):
                return True
        return False


# Implementacja algorytmu minmax
def min_max(node, deep, turn, chosen_nodes=[]):
    w = []
    if((deep == 0) | node.terminal()):
        return [node, node.h()]
    U = node.successors
    for u in U:
        w.append(min_max(u, deep - 1, not(turn), chosen_nodes))
    if(turn):
        max = max_rand(w)
        chosen_nodes.append(max[0].name)
        return max
    else:
        min = min_rand(w)
        chosen_nodes.append(min[0].name)
        return min


# Funkcja generująca drzewo gry kółko i krzyżyk z zadanego stanu i z ogreśloną głębokością
def tic_tac_toe_tree(node, turn, deep):
    if(deep > 0):
        for i in range(0, 3, 1):
            for j in range(0, 3, 1):
                if(node.board[i][j] == 0):
                    new_board = [[0 for _ in range(3)] for _ in range(3)]
                    for x in range(0, 3, 1):
                        for y in range(0, 3, 1):
                            new_board[x][y] = node.board[x][y]
                    if(turn):
                        new_board[i][j] = 1
                    else:
                        new_board[i][j] = -1
                    node.add_successor(NodeTicTacToe([], new_board, id(new_board)))
        if not(node.terminal()):
            for successor in node.successors:
                tic_tac_toe_tree(successor, not(turn), deep - 1)
        else:
            node.successors = []


# Funkcja znajdująca i prezentująca najlepszy kolejny ruch
def next_move(node, turn, deep):
    tic_tac_toe_tree(node, turn, deep)
    chosen_nodes = []
    min_max(node, deep, turn, chosen_nodes)
    node.print_graph(chosen_nodes, turn)


# Funkcja zwracająca największy element tablicy pod względem wartości z założeniem losowania elementów równie dobrych
def max_rand(array):
    max = array[0]
    for elem in array:
        if(elem[1] > max[1]):
            max = elem
        elif(elem[1] == max[1]):
            if(random.uniform(0, 1) >= 0.5):
                max = elem
    return max


# Funkcja zwracająca najmniejszy element tablicy pod względem wartości z założeniem losowania elementów równie dobrych
def min_rand(array):
    min = array[0]
    for elem in array:
        if(elem[1] < min[1]):
            min = elem
        elif(elem[1] == min[1]):
            if(random.uniform(0, 1) >= 0.5):
                min = elem
    return min


# Funkcja konfugurowalnego uruchomienia
def manual_run():
    board = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for i in range(0, 3, 1):
        for j in range(0, 3, 1):
            print(f'Insert value -1 for "X" 1 for "O" or 0 for empty on position ({i},{j})')
            board[i][j] = int(input())

    print('depth')
    depth = int(input())

    print('max - 1, min - 2 ')
    turn = int(input())

    node = NodeTicTacToe([], board, id(board))
    next_move(node, True, depth) if turn == 1 else next_move(node, False, depth)


# Funkcja realizująca uruchomiee dla wybranych wartości planszy początkowej
def auto_run_test():
    board = [[1, 0, -1], [0, 0, 0], [-1, 0, 1]]
    node = NodeTicTacToe([], board, id(board))
    next_move(node, False, 2)
    node.successors = []
    next_move(node, False, 3)
    node.successors = []
    next_move(node, False, 4)

def main():
    print('Chose run mode \n1-for manual \n2-for auto test')
    runMode = int(input())
    if (runMode == 1):
        manual_run()
    elif (runMode == 2):
        auto_run_test()
    else:
        print('wrong input')


if __name__ == "__main__":
    main()


