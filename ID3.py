import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from collections import Counter
import networkx as nx
import pandas as pd
from networkx.drawing.nx_pydot import graphviz_layout
import random
import sys

sys.setrecursionlimit(1500)

# **************************************************************
# * Wykonał : Adrian Pruszyński                                *
# * Nr. albumu: 274424                                         *
# * EITI, Informatyka, 2021L                                   *
# * WSI LAB 6                                                  *
# **************************************************************


#  Klasa na bazie której definiowane jest drzewo
class Tree:
    def __init__(self, successors, name, connection_name="", val=None):
        self.successors = successors
        self.name = str(name) + '\n' + str(counter())
        self.connection_name = str(connection_name)
        self.val = val

# Funkcja odpowiadająca za dodawanie następników
    def add_successor(self, successor, connection_name=""):
        if(connection_name != ""):
            successor.connection_name = str(connection_name)
        self.successors.append(successor)

# Funkcja odpowiadająca za wyświetlenie grafu
    def print_graph(self, chosen_nodes=[], info=None):
            connection_from = []
            connections_to = []
            connections_names = {}

            self.connections(connection_from, connections_to, connections_names)
            df = pd.DataFrame({'from': connection_from, 'to': connections_to})

            G = nx.from_pandas_edgelist(df, 'from', 'to')
            pos = graphviz_layout(G, prog="dot")

            for e in G.edges():
                G[e[0]][e[1]]['color'] = 'black'
            if (len(chosen_nodes) > 0):
                p = nx.shortest_path(G, self.name, chosen_nodes[len(chosen_nodes) - 1])
                for i in range(len(p) - 1):
                    G[p[i]][p[i + 1]]['color'] = 'magenta'
            edge_color_list = [G[e[0]][e[1]]['color'] for e in G.edges()]
            color_map = []

            for node in G:
                if str(node).find("Pewna = ") != -1:
                    color_map.append('greenyellow')
                elif str(node).find("Najczęstsza = ") != -1:
                    color_map.append('tan')
                else:
                    color_map.append('cornflowerblue')

            red_patch = mpatches.Patch(color='greenyellow', label='Pewna')
            green_patch = mpatches.Patch(color='tan', label='Najczęściej występująca')
            plt.legend(handles=[red_patch, green_patch])
            if(info != None):
                plt.title(info)
            nx.draw(G, node_color=color_map, edge_color=edge_color_list, font_size=7, with_labels=True, pos=pos)
            nx.draw_networkx_edge_labels(G, pos, edge_labels=connections_names, font_size=7, font_color='red')
            plt.show()

# Funkcja odpowiadająca za utwożenie listy połączeń w grafie
    def connections(self, connection_from, connections_to, connections_names):
            for successor in self.successors:
                connection_from.append(self.name)
                connections_to.append(successor.name)
                connections_names[self.name, successor.name] = successor.connection_name
                successor.connections(connection_from, connections_to, connections_names)

# Funckja odpowiedzialna za predykcję
    def predict(self, record):
        chosen_nodes = []
        walker = self
        while(True):
            next = [succesor for succesor in walker.successors if str(succesor.connection_name) == str(record[walker.name.split('\n')[0]])]
            if (next == []):
                return walker.val, chosen_nodes
            chosen_nodes.append(next[0].name)
            walker = next[0]

# Funckja odpowiedzialna za obliczenie acc
    def acc(self,Y, data):
        true_predictions = 0
        all_predictions = 0
        for index in data.index.values:
            val, _ = self.predict(data.loc[index])
            all_predictions += 1
            if(val == data.loc[index][Y]):
                true_predictions += 1
        return true_predictions/all_predictions


# Counter wykorzytywany do rysowania drzewa
def counter(init=[0]):
    init[0] += 1
    return init[0]


# Funkcja realizująca algorytm ID#
def ID3(Y, D, U, deep=1000):
    deep -= 1
    if(all_has_same_class(U, Y)):
        val = U[Y].values[0]
        return Tree([], "Pewna = " + str(val), val=val)
    if((len(D) == 0) | (deep == 0)):
        val = frequent_class(U, Y)
        return Tree([], "Najczęstsza = " + str(val), val=val)
    d = arg_max_infGain(D, U, Y)
    Uj = split_by_attribute(d, U)
    node = Tree([], d)
    D.remove(d)
    for uj in Uj:
        node.add_successor(ID3(Y, list(D), uj[1], deep), uj[0])
    return node


#Funkcja odpowiedzialna za podział danych względem atrybutu
def split_by_attribute(d, U):
    Uj = []
    for p, k in U.groupby([d]):
        k_drop = k.drop([d], axis=1)
        if(len(k_drop) != 0):
            Uj.append([p, k_drop])
    return Uj


# Funkcja sprawdzajaca czy cały zbiór ma tę samą wartość klasy Y
def all_has_same_class(U, Y):
    first_value = U[Y].values[0]
    return all(value == first_value for value in U[Y])


# funkcja zwracająca najczestrza klase jesli istnieją klasy równo częste zwraca losową traktując je jako tak samo dobre
def frequent_class(U, Y):
    values_list = U[Y].values
    random.shuffle(values_list)
    return Counter(values_list).most_common()[0][0]


# Funkcja wyliczająca argmaxinfGain
def arg_max_infGain(D, U, Y):
    infGain_list = [[infGain(d, U, Y), d] for d in D]
    random.shuffle(infGain_list)
    return max(infGain_list, key=lambda item: item[0])[1]


#Funkcja wyliczająca infGain
def infGain(d, U, Y):
    return I(U, Y) - Inf(d, U, Y)


# Funkcja wyliczająca entropię
def I(U, Y):
    val = 0
    unique_values = U[Y].unique()
    for u_val in unique_values:
        f = U[Y].value_counts()[u_val] / len(U[Y])
        val += -f * np.log2(f)
    return val


# Funkcja wyliczająca enropię zbioru porzielonego na podzbiory atrybutem d
def Inf(d, U, Y):
    Uj = split_by_attribute(d, U)
    val = 0
    for uj in Uj:
        val += len(Uj)/len(U) * I(uj[1], Y)
    return val


# Funkcja wczytująca oraz obrabiająca dane
def read_data(age_interval_num=5, fare_interval_num=5, age_interval=None, fare_interval=None, cust_intervals=False):
    df = pd.read_csv("titanic.csv")
    pd.set_option("display.max_rows", None, "display.max_columns", None)

    df = df.drop(['Name'], axis=1)
    if(cust_intervals):
        df["Age"] = pd.cut(df["Age"], bins=age_interval)
        df["Fare"] = pd.cut(df["Fare"], bins=fare_interval, right=False)
    else:
        df["Age"] = pd.cut(df["Age"], age_interval_num, precision=2)
        df["Fare"] = pd.cut(df["Fare"], fare_interval_num, right=False, precision=2)

    return df, list(df.columns)


def manual_run():
    print('Podaj maksymalną głębokość')
    deep = int(input())
    print('Podaj ilość podziałów dla "Age"')
    age_int = int(input())
    print('Podaj ilość podziałów dla "Fare"')
    fare_int = int(input())

    U, D = read_data(age_int, fare_int)
    train, validate, test = np.split(U.sample(frac=1, random_state=42), [int(.6 * len(U)), int(.8 * len(U))])
    tree = ID3(D[0], D[1:], train, deep)

    print('val acc = ', tree.acc(D[0], validate))
    print('test acc = ', tree.acc(D[0], test))
    tree.print_graph()


# Funkcja realizująca uruchomienie testowe
def auto_run_test():
    age_interval = pd.IntervalIndex.from_tuples([
        (0, 6),
        (6, 12),
        (12, 16),
        (16, 25),
        (25, 35),
        (35, 45),
        (45, 60),
        (60, 400)
    ], closed='left')
    fare_interval = pd.IntervalIndex.from_tuples([
        (0, 15),
        (15, 25),
        (25, 60),
        (60, 100),
        (100, 150),
        (150, 300),
        (300, 1000)
    ], closed='left')

    intervals = [[8, 8], [8, 4], [4, 8], [4, 4]]
    deep_arr = [2, 3, 5, 7]
    for deep in deep_arr:
        print('Custom intervals')
        print('deep = ', deep)
        U, D = read_data(age_interval=age_interval, fare_interval=fare_interval, cust_intervals=True)
        train, validate, test = np.split(U.sample(frac=1, random_state=42), [int(.6 * len(U)), int(.8 * len(U))])
        tree = ID3(D[0], D[1:], train, deep)
        print('acc = ',tree.acc(D[0], validate))
        for interval in intervals:
            print(interval[0], ' - ', interval[1])
            print('deep = ', deep)
            U, D = read_data(interval[0], interval[1])
            train, validate, test = np.split(U.sample(frac=1, random_state=42), [int(.6 * len(U)), int(.8 * len(U))])
            tree = ID3(D[0], D[1:], train, deep)
            print('acc =', tree.acc(D[0], validate))


    # val, chosen_nodes = tree.predict(validate.loc[validate.index.values[6]])

    # tree.print_graph(chosen_nodes, str(dict(validate.loc[validate.index.values[6]])))
    tree.print_graph()


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


