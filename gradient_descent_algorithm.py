import math
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# **************************************************************
# * Wykonał : Adrian Pruszyński                                *
# * EITI, Informatyka, 2021L                                   *
# **************************************************************

last_index_number = 0


# Funkcja konfugurowalnego uruchomienia
def manual_run():
    print('Function f1 or f2 by press 1 or 2')
    function = int(input('f'))

    print('Step size')
    step_size = float(input())

    print('Max number of executed steps')
    max_steps_number = int(input())

    print('Epsilon')
    epsilon = float(input())

    if (function == 1):
        print('Start point x')
        input_x = float(input('x = '))
        start_point = np.array([input_x])
        create_files()
        gradient_descent(step_size, max_steps_number, epsilon, start_point, f1, f1_gradient)
    elif (function == 2):
        print('Start point x1')
        input_x1 = float(input('x1 = '))
        print('Start point x2')
        input_x2 = float(input('x2 = '))
        print('Start point (', input_x1, ', ', input_x2, ')')
        start_point = np.array([input_x1, input_x2])
        create_files()
        gradient_descent(step_size, max_steps_number, epsilon, start_point, f2, f2_gradient)
    else:
        print('wrong input')


# Funkcja realizująca serię uruchomień dla wybranych wartości punktów startowych oraz wielkości kroku
def auto_run_test():
    create_files()

    test_step_sizes = [0.8, 0.1, 0.01, 0.001]
    test_start_points_f1 = [[3], [-49], [50], [-500], [1000]]
    test_start_points_f2 = [[3, 6], [-49, -53], [100, -30], [-500, 800], [1000, 1000]]

    for start_point in test_start_points_f1:
        for step in test_step_sizes:
            gradient_descent(step, 50000, 0.001, np.array(start_point), f1, f1_gradient)

    for start_point in test_start_points_f2:
        for step in test_step_sizes:
            gradient_descent(step, 50000, 0.001, np.array(start_point), f2, f2_gradient)


# Funkcja realizująca algorytm najszybszego spadku
def gradient_descent(step_size, max_steps_number, epsilon, start_point, f, f_grad):
    i = 0
    actual_point = start_point
    print('step ', i, '|actual point ', actual_point, '|grad ', step_size * f_grad(actual_point), '|val ', f(actual_point))
    points = [[i, actual_point, f(actual_point)]]

    while(np.any(np.abs(f_grad(actual_point)) >= epsilon) & (i < max_steps_number)):
        actual_point = actual_point - step_size * f_grad(actual_point);
        i = i + 1
        points.append([i, actual_point, f(actual_point)])
        print('step ', i, '|point ', actual_point, '|grad ', step_size * f_grad(actual_point), '|val ', f(actual_point))

    if not (i < max_steps_number):
        print('Reached max number of executed steps')

    results(points, step_size, epsilon, start_point)


# Funkcja odpowiedzialna za prezentację i zapis rezultatów
def results(data, step_size, epsilon, start_point):
    step = []
    function_value = []
    points = []
    x = []
    y = []

    for row in data:
        step.append(row[0])
        points.append(row[1])
        function_value.append(row[2])

    if (points[0].shape[0] == 2):
        title = f"step={step_size} epsilon={epsilon} start point=({start_point[0]},{start_point[1]})"

        for point in points:
            x.append(point[0])
            y.append(point[1])

        plt.xlabel('step')
        plt.ylabel('f(x)')
        plt.title('f2 ' + title)
        plt.scatter(step, function_value, s=5)

        plt.savefig(f"results\\graphs\\f2 {title} step f(x).png")
        plt.show()

        stop_point = points.pop()
        f = open('results\\data\\f2.csv', 'a')
        np.savetxt(f,
                   [[step_size, epsilon, start_point[0], start_point[1], function_value.pop(), step.pop(), stop_point[0], stop_point[1]]],
                   delimiter="; ",
                   fmt='% s', )
        f.close()

    elif(points[0].shape[0] == 1):
        title = f"step={step_size} epsilon={epsilon} start point={start_point[0]}"

        for point in points:
            x.append(point[0])

        plt.scatter(x, function_value, s=10)
        plt.plot(x, function_value, ':', c='red')
        plt.xlabel('x')
        plt.ylabel('f(x)')
        plt.title('f1 ' + title)

        plt.savefig(f"results\\graphs\\f1 {title} x f(x).png")
        plt.show()

        plt.xlabel('step')
        plt.ylabel('f(x)')
        plt.title('f1 ' + title)
        plt.scatter(step, function_value, s=10)

        plt.savefig(f"results\\graphs\\f1 {title} step f(x).png")
        plt.show()

        f = open('results\\data\\f1.csv', 'a')
        np.savetxt(f,
                   [[step_size, epsilon, start_point, function_value.pop(), step.pop(), points.pop()[0]]],
                   delimiter="; ",
                   fmt='% s', )
        f.close()


# Funkcja odpowiedzialna za utworzenie i inicjalizację wykożystywanych plików oraz katalogów
def create_files():
    Path("results\\graphs").mkdir(parents=True, exist_ok=True)
    Path("results\\data").mkdir(parents=True, exist_ok=True)

    f = open('results\\data\\f1.csv', 'w')
    np.savetxt(f,
               [["wielkosc_kroku", "epsilon", "punkt_poczatkowy", "f(x)", "kroki", "punkt_koncowy_x"]],
               delimiter="; ",
               fmt='% s', )
    f.close()

    f = open('results\\data\\f2.csv', 'w')
    np.savetxt(f,
               [["wielkosc_kroku", "epsilon", "punkt_poczatkowy_x1", "punkt_poczatkowy_x2", "f(x)", "kroki", "punkt_koncowy_x1", "punkt_koncowy_x2"]],
               delimiter="; ",
               fmt='% s', )
    f.close()


# Funkcja obliczająca wartość f1(x)
def f1(x): return x ** 2


# Funkcja obliczająca wartość f2(x1, x2)
def f2(x, a=last_index_number): return (x[0] + a) ** 2 + (x[1] - a) ** 2 - 5 * math.cos(10 * math.sqrt((x[0] + a) ** 2 + (x[1] - a) ** 2))


# Funkcja obliczająca wartość gradientu f1(x)
def f1_gradient(x): return 2 * x


# Funkcja obliczająca wartość gradientu f2(x1, x2)
def f2_gradient(x, a=last_index_number): return np.array([(x[0] + a) * (2 + g(x[0], x[1], a)), (x[1] - a) * (2 + g(x[0], x[1], a))])


# Funkcja obliczająca składową wartość wykorzystywaną w obliczaniu gradientu f2(x1, x2)
def g(x1, x2, a): return 50 * math.sin(10 * math.sqrt((x1 + a) ** 2 + (x2 - a) ** 2)) / math.sqrt((x1 + a) ** 2 + (x2 - a) ** 2)


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




