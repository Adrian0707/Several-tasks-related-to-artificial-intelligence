import random
import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import time

# **************************************************************
# * Wykonał : Adrian Pruszyński                                *
# * EITI, Informatyka, 2021L                                   *
# **************************************************************

# Maksymalna wartość liczby wektorze osobnika
max_xi = 3
# Minimlna wartość liczb wektorze osobnika
min_xi = -4


# Funkcja generująca losową populację zadanej liczności
def random_population(population_size):
    population = []
    for i in range(0, population_size, 1):
        population.append([random.randint(min_xi, max_xi),
                           random.randint(min_xi, max_xi),
                           random.randint(min_xi, max_xi),
                           random.randint(min_xi, max_xi),
                           random.randint(min_xi, max_xi),
                           random.randint(min_xi, max_xi)])
    return population


# Funkcja realizującą algorytm genetyczny w wersji Hollanda
def genetic_algorithm(q_x, start_population, population_size, mutation_probability, crossover_probability, max_iterations):
    start_time = time.time()
    t = 0
    scores = score(start_population, q_x)
    best_prev_gen, bests_score_prev_gen = find_best(scores, start_population)
    population = start_population
    data_scores = [scores]
    data_best = []
    data_best.append([best_prev_gen, bests_score_prev_gen, best_prev_gen, bests_score_prev_gen])
    while(t < max_iterations):
        to_reproduction = reproduction(population, scores, population_size)
        population = crossover_and_mutation(to_reproduction, mutation_probability, crossover_probability)
        scores = score(population, q_x)
        best_next_gen, bests_score_next_gen = find_best(scores, population)
        if(bests_score_next_gen >= bests_score_prev_gen):
            best_prev_gen = best_next_gen
            bests_score_prev_gen = bests_score_next_gen
        t = t + 1
        data_scores.append(scores)
        data_best.append([best_prev_gen, bests_score_prev_gen, best_next_gen, bests_score_next_gen])

    results(data_scores, data_best, population_size, mutation_probability, crossover_probability, round(time.time() - start_time, 2))
    return bests_score_prev_gen, best_prev_gen


# Funkcja odpowiedzialna za prezentację i zapis rezultatów
def results(data_scores, data_best, population_size, mutation_probability, crossover_probability, time):
    title = f"population size={population_size} mutation probability={mutation_probability} " \
        f"crossover probability={crossover_probability} time={time}"

    plt.rcParams["figure.figsize"] = (20, 10)
    plt.xlabel('step')
    plt.ylabel('f(x)')
    plt.title(title)


    iteration = np.array(range(0, len(data_scores), 1))
    iteration2 = []
    for i in iteration:
        for j in range(0, population_size, 1):
            iteration2.append(i)

    data_scores = np.array(data_scores).reshape(-1)
    data_best = np.transpose(np.array(data_best, dtype=np.object))

    plt.scatter(iteration2, data_scores, s=2, label='Inne')
    plt.scatter(iteration, data_best[1], s=5, c='red', label='Najlepszy znaleziony')
    plt.scatter(iteration, data_best[3], s=3, c='green', label='Najlepszy w populacji')
    plt.legend()
    plt.savefig(f"results\\graphs\\graph {title} step f(x).png")
    plt.clf()

    f = open('results\\data\\data.csv', 'a')
    np.savetxt(f, [[len(data_best[1]) - 1, population_size, mutation_probability, crossover_probability, data_best[1][-1],
                    data_best[0][-1], time]],
               delimiter="; ", fmt='% s', )
    f.close()
    print(title)


# funkcja realizująca ocenę przystosowania na bazie funkcji celu z zabezpieczeniem przed wystąpieniem liczb ujemnych
def score(population, q_x):
    scores = []
    for individual in population:
        val = q_x(individual)
        if(val < 0):
            scores.append(0)
        elif(val >= 0):
            scores.append(val)
    return np.array(scores).astype(int)


# funkcja znajdująca nalepszego osobnika oraz jego ocenę
def find_best(scores, population):
    max_score_index = np.argmax(scores)
    return population[max_score_index], scores[max_score_index]


# funkcja wybierająca osobniki do reprodukcji
def reproduction(population, scores, population_size):
    probabilities = []
    to_reproduction = []
    sum_scores = sum(scores)
    for score in scores:
        probabilities.append(score/sum_scores)
    for _ in range(population_size):
        random_val = random.random()
        probability_sum = 0
        i = 0
        for probability in probabilities:
            probability_sum += probability
            if(probability_sum >= random_val):
                to_reproduction.append(population[i])
                break
            i = i + 1
    return to_reproduction


# funkcja realizująca krzyżowanie oraz mutację
def crossover_and_mutation(reproduction, mutation_probability, crossover_probability):
    chromosomes = to_binary_chromosomes(reproduction)
    for i in range(0, len(chromosomes), 2):
        if(crossover_probability >= random.random()):
            if(i+1 < len(chromosomes)):
                rand_val = random.randint(1, 23)
                child_1 = np.concatenate((chromosomes[i][0: rand_val], chromosomes[i + 1][rand_val:]), axis=0)
                child_2 = np.concatenate((chromosomes[i + 1][0: rand_val], chromosomes[i][rand_val:]), axis=0)
                chromosomes[i] = child_1
                chromosomes[i+1] = child_2

    for chromosome in chromosomes:
        for i in range(0, len(chromosome), 1):
            if (mutation_probability >= random.random()):
                chromosome[i] = math.fabs(chromosome[i] - 1)

    # Przygotowanie funkcji sprowadzania wartości do zakresu dla wektorów
    vector_values_to_range = np.vectorize(values_to_range)
    return vector_values_to_range(to_numeric_vectors(chromosomes))


#funkcja sprowadzająca wartoście do zakresu <min_xi; max_xi>
def values_to_range(v):
    if(v > max_xi):
        v = max_xi
    elif(v < min_xi):
        v = min_xi
    return v


# Funkcja odpowiedzialna za utworzenie i inicjalizację wykożystywanych plików oraz katalogów
def create_files():
    Path("results\\graphs").mkdir(parents=True, exist_ok=True)
    Path("results\\data").mkdir(parents=True, exist_ok=True)

    f = open('results\\data\\data.csv', 'w')
    np.savetxt(f,
               [["steps", "population_size", "mutation_probability", "crossover_probability", "bests_score",
                 "best_individual", "time"]],
               delimiter="; ",
               fmt='% s', )
    f.close()


#Funkcja która zostaje poddana maksumalizacji
def q(x):
    sum = 0
    for xi in x:
        sum += xi ** 4 - 16 * xi ** 2 + 5 * xi
    return - sum / 2


# funkcja zamieniająca wektory liczbowe osobników na ich chromosomy
def to_binary_chromosomes(x):
    # funkcja pozwalająca na zamianę wektora z wartościami liczbowymi na zawierający reprezentacje binarną w postaci stringów
    number_v_to_binary_string_v = np.vectorize(np.binary_repr)

    # zamiana wektora na binarny oraz zapis w postaci np.array przechowującej stringi o długości 4
    binary_string_array = np.array(number_v_to_binary_string_v(x, width=4), dtype='|S4')

    # zamiana wartości stringowych na numeryczne odpowiadające znakom ASCII oraz zmiana kształtu na odpowiedni dla chromosomu
    binary_numeric_array = (np.frombuffer(binary_string_array, dtype=np.byte)).reshape(-1, np.shape(x)[1] * 4)

    #odjęcie od kazdej wartości w tablicy 48, gdyż wartość "0" w tablicy ASCII równa się 48 a "1" równa się 49
    return binary_numeric_array - 48


# funkcja zamieniająca chromosomy osobników na ich wektory liczbowe
def to_numeric_vectors(x):
    # Zmiana krztałtu na odpowiadajacy rerprezentacji bitowej kolejnych liczb w wektorach dla kolejnych osobników
    x = x.reshape(-1, 6, 4)
    # Mnożenie powstałej macierzy przez macierz wartości kolejnych bitów
    return np.matmul(x, [-8, 4, 2, 1])


# Funkcja konfugurowalnego uruchomienia
def manual_run():
    create_files()

    print('Max number of executed steps')
    max_steps_number = int(input())

    print('Population size')
    population_size = int(input())

    print('Mutation probability')
    mutation_probability = float(input())

    print('Crossover probability')
    crossover_probability = float(input())

    genetic_algorithm(q, random_population(population_size), population_size, mutation_probability, crossover_probability, max_steps_number)


# Funkcja realizująca serię uruchomień dla wybranych wartości punktów startowych oraz wielkości kroku
def auto_run_test():
    create_files()

    population_sizes = [200, 100, 50, 10, 5]
    mutation_probabilities = [0.1, 0.01, 0.001, 0.0001]
    crossover_probabilities = [0.9, 0.5, 0.1, 0.01]
    population_sizes.reverse()

    for population_size in population_sizes:
        for mutation_probability in mutation_probabilities:
            for crossover_probability in crossover_probabilities:
                genetic_algorithm(q, random_population(population_size), population_size, mutation_probability,
                                  crossover_probability, 10000)


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


