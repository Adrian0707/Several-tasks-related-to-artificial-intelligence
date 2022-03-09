import numpy as np
import random
from pathlib import Path
import matplotlib.pyplot as plt

# **************************************************************
# * Wykonał : Adrian Pruszyński                                *
# * EITI, Informatyka, 2021L                                   *
# **************************************************************

N = 4

attributes1 = ["1. Alcohol",
              "2. Malic acid",
              "3. Ash",
              "4. Alcalinity of ash",
              "5. Magnesium",
              "6. Total phenols",
              "7. Flavanoids",
              "8. Nonflavanoid phenols",
              "9. Proanthocyanins",
              "10. Color intensity",
              "11. Hue",
              "12. OD280/OD315",
              "13. Proline"]


# Funkcja realizująca podział danych na podzbiory ze względu na klasy
def separate_classes(x_train, y_train):
    separated_classes = {}
    for i in range(len(x_train)):
        feature = x_train[i]
        class_name = y_train[i]
        if class_name not in separated_classes:
            separated_classes[class_name] = []
        separated_classes[class_name].append(feature)
    return separated_classes


# Funkcja odpowiedzialna za wyliczanie Śr oraz Odch. standardowego
def stat_info(feature):
    stat = []
    for feature in np.transpose(feature):
        stat.append([np.std(feature), np.mean(feature)])
    return stat


# Funkcja odpowiedzialna za przygotowanie słownika z wartościami P(A), Śr oraz odchylenia standardowego
def fit(x_train, y_train):
    separated_classes = separate_classes(x_train, y_train)
    class_summary = {}
    for class_name, feature in separated_classes.items():
        class_summary[class_name] = [
             len(feature) / len(x_train),
             [i for i in stat_info(feature)]
        ]
    return class_summary


# Funcka odpowiedzialna za wyliczenie wartości rozkładu normalnego
def normal_distribution(feature, mean, std):
    exponent = np.exp(-((feature - mean) ** 2 / (2 * std ** 2)))
    return exponent / (np.sqrt(2 * np.pi) * std)


# Funkcja odpowiedzialna za predykcję klas
def predict(data, class_summary):
    predicted = []
    for row in data:
        joint_proba = {}
        for class_name, features in class_summary.items():
            total_features = len(features[1])
            p_a_b = 1
            for i in range(total_features):
                feature = row[i]
                mean = features[1][i][1]
                stdev = features[1][i][0]
                normal_proba = normal_distribution(feature, mean, stdev)
                p_a_b *= normal_proba
            p_a = features[0]
            joint_proba[class_name] = p_a * p_a_b
        predicted.append(max(joint_proba, key=joint_proba.get))
    return predicted


# Funkcja liczaca dokładność klasyfikacji
def accuracy(y_test, y_pred):
    positive = 0
    for y_t, y_p in zip(y_test, y_pred):
        if y_t == y_p:
            positive += 1
    return positive / len(y_test)


# Funkcja wczytująca zbiór danych
def data_load():
    file = open('wine.data', 'r')
    data = []
    for line in file:
        data.append(line)
    file.close()
    data = [i.strip() for i in data]

    split_data = [d.split(",") for d in data]

    return list(np.array(split_data).astype(float))


# Funkcja realizująca walidację krzyżową oraz dzieląca zbiór danych na podzbiory
def cross_validation_split(data, folds):
    data_test = list()
    data_copy = list(data)
    fold_size = int(len(data) / folds)
    for i in range(folds):
        fold = list()
        data_train = [list(data)] * folds
        while len(fold) < fold_size:
            index = random.randrange(len(data_copy))
            fold.append(data_copy.pop(index))
            data_train[i].pop(index)
        data_test.append(fold)

    x_test, y_test = np.transpose(np.transpose(data_test)[1:]), np.transpose(np.transpose(data_test)[0])
    x_train, y_train = np.transpose(np.transpose(data_train)[1:]), np.transpose(np.transpose(data_train)[0])

    return x_test, y_test, x_train, y_train


# Funkcja licząca średnią dokładność klasyfikacji przy użyciu walidacji krzyżowej
def cross_validation_accuracy_test(data, folds):
    x_test, y_test, x_train, y_train = cross_validation_split(data, folds)

    avg_accuracy = 0
    for i in range(folds):
        class_summary = fit(x_train[i], y_train[i])
        y_pred = predict(x_test[i], class_summary)
        avg_accuracy += accuracy(y_test[i], y_pred)
    avg_accuracy /= folds
    return avg_accuracy


# Funkcja opowiedzialna za znalezienie najlepszego atrybutu pod wzgędem dokładności klasyfikacji
def find_best_accuracy_attribute(data, folds):
    accuracy_arr = []
    max_acc = 0
    attribute = None
    for i in range(len(attributes1)):
        new_arr = np.transpose([np.transpose(data)[0], np.transpose(data)[i+1]])
        avg_acc = cross_validation_accuracy_test(new_arr, folds)
        accuracy_arr.append(avg_acc)

        if(avg_acc >= max_acc):
            max_acc = avg_acc
            attribute = i + 1
    results_find_best_accuracy_attribute(accuracy_arr, attribute, max_acc, folds)
    return attribute


# Funkcja odpowiedzialna za prezentację i zapis rezultatów
def results_find_best_accuracy_attribute(data_scores, attribute, max_acc, folds):
    title = f"Dokładność klasyfikacji dla walidacji krzyżowej gdzie n={folds} "

    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = (10, 5)
    fig, ax = plt.subplots()
    plt.xlabel('Trafność')
    plt.ylabel('Atrybuty')
    plt.title(title)
    ax.barh(attributes1, data_scores, align='center')
    ax.set_yticks(attributes1)
    ax.set_yticklabels(attributes1)
    fig.tight_layout()
    plt.savefig(f"results\\graphs\\graph {title} .png")
    plt.clf()

    f = open('results\\data\\best_attribute.csv', 'a')
    np.savetxt(f, [[folds, attributes1[attribute - 1], max_acc]],
               delimiter="; ", fmt='% s', )
    f.close()


# Funkcja opowiedzialna za dodanie atrybutu oraz porównanie zmian jakie zaszły w zakresie dokładności klasyfikacji
def add_atribute_and_compare(best_accuracy_attribute, data, folds):
    new_attribute = N + 1

    if(best_accuracy_attribute == new_attribute):
        new_attribute += 1

    new_arr_with_added_attribute = np.transpose([np.transpose(data)[0], np.transpose(data)[best_accuracy_attribute], np.transpose(data)[new_attribute]])
    new_arr = np.transpose([np.transpose(data)[0], np.transpose(data)[best_accuracy_attribute]])
    best_attribute_accuracy = cross_validation_accuracy_test(new_arr, folds)
    best_attribute_accuracy_with_extra_attribute_added = cross_validation_accuracy_test(new_arr_with_added_attribute, folds)

    results_add_atribute_and_compare(new_attribute, best_accuracy_attribute,
                                     best_attribute_accuracy, best_attribute_accuracy_with_extra_attribute_added, folds)


# Funkcja odpowiedzialna za prezentację i zapis rezultatów
def results_add_atribute_and_compare(added_attribute, best_accuracy_attribute ,best_attribute_accuracy,
                                     best_attribute_accuracy_with_extra_attribute_added, folds):
    title = f"Porównanie dokładności klasyfikacji gdzie n={folds} "

    plt.rcdefaults()
    plt.rcParams["figure.figsize"] = (10, 5)
    fig, ax = plt.subplots()
    plt.xlabel('Trafność')
    plt.title(title)
    label = [attributes1[best_accuracy_attribute - 1], attributes1[added_attribute - 1] + " oraz " + attributes1[best_accuracy_attribute - 1]]
    ax.barh(label, [best_attribute_accuracy, best_attribute_accuracy_with_extra_attribute_added], align='center')
    ax.set_yticks(label)
    ax.set_yticklabels(label)
    fig.tight_layout()

    plt.savefig(f"results\\graphs\\graph {title} .png")
    plt.clf()

    f = open('results\\data\\compare_attributes.csv', 'a')
    np.savetxt(f, [[folds, attributes1[best_accuracy_attribute - 1], attributes1[added_attribute - 1],
                    best_attribute_accuracy, best_attribute_accuracy_with_extra_attribute_added]],
               delimiter="; ", fmt='% s', )
    f.close()


# Funkcja odpowiedzialna za utworzenie i inicjalizację wykożystywanych plików oraz katalogów
def create_files():
    Path("results\\graphs").mkdir(parents=True, exist_ok=True)
    Path("results\\data").mkdir(parents=True, exist_ok=True)

    f = open('results\\data\\best_attribute.csv', 'w')
    np.savetxt(f,
               [["folds", "attribute", "avg_acc"]],
               delimiter="; ",
               fmt='% s', )
    f.close()

    f = open('results\\data\\compare_attributes.csv', 'w')
    np.savetxt(f,
               [["folds", "best_attribute", "added_attribute", "avg_acc_best", "avg_acc_best_with_added"]],
               delimiter="; ",
               fmt='% s', )
    f.close()


def main():
    create_files()
    data = data_load()
    folds_arr = [2, 5, 10, 15, 20, 30]
    for folds in folds_arr:
        best_acccuracy_attribute = find_best_accuracy_attribute(data, folds)
        add_atribute_and_compare(best_acccuracy_attribute, data, folds)

if __name__ == "__main__":
    main()
