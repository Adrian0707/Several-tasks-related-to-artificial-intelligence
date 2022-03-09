import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from tensorflow.keras.datasets import mnist
import time
from pathlib import Path
import itertools
from sklearn.model_selection import train_test_split

# **************************************************************
# * Wykonali : Anna Kudzia oraz Adrian Pruszyński              *
# * EITI, Informatyka, 2021L                                   *
# **************************************************************


class Model:
    def __init__(self, layers_dims, learning_rate, epochs, batch_size):
        self.params = {}  # dictionary containing parameters
        self.num_layers = len(layers_dims) # number of layers without input layer
        self.layers_dims = layers_dims  # layers dimenstions
        self.learning_rate = learning_rate  # the learning rate
        self.num_epochs = epochs  # number of epochs
        self.grads = {}  # dictionary with the gradients
        self.batch_size = batch_size # mini batch size

    def sigmoid(self, x):
        # activation function of output layer
        s = 1 / (1 + np.exp(-x))
        return s

    def sigmoid_derivative(self, Z):
        s = 1 / (1 + np.exp(-Z))
        return s * (1 - s)

    def relu(self, x):
        # activation function of hidden layer
        s = np.maximum(0, x)
        return s

    def d_relu(self, x):
        # derivative of relu activation function
        return (x > 0) * 1

    def softmax(self, x):
        # activation function of output layer, simplified version to prevent overflow
        e = np.exp(x - x.max())
        return e / np.sum(e, axis=0)

    def initialize_parameters(self):
        for l in range(1, self.num_layers):
            self.params['W' + str(l)] = np.random.randn(self.layers_dims[l], self.layers_dims[l - 1]) / np.sqrt( self.layers_dims[l - 1])
            self.params['b' + str(l)] = np.zeros((self.layers_dims[l], 1))

    def cross_entropy(self, Y, A):
        # cost function for multi class classification problem - categorical cross entropy
        return -np.mean(Y * np.log(A.T + 1e-8))

    def create_minibatches(self, X, y):
        mini_batches = []
        data = np.hstack((X, y))
        np.random.shuffle(data)
        n_minibatches = data.shape[0] // self.batch_size
        for i in range(1, n_minibatches + 1):
            mini_batch = data[i * self.batch_size:(i + 1) * self.batch_size, :]
            X_mini = mini_batch[:, :-10]
            Y_mini = mini_batch[:, -10:].reshape((-1, 10))
            mini_batches.append((X_mini, Y_mini))
        if data.shape[0] % self.batch_size != 0:
            mini_batch = data[i * self.batch_size:data.shape[0]]
            X_mini = mini_batch[:, :-10]
            Y_mini = mini_batch[:, -10:].reshape((-1, 10))
            mini_batches.append((X_mini, Y_mini))
        return mini_batches

    def forward_propagation(self, X):
        store = {}

        A = X.T
        for l in range(self.num_layers - 2):
            Z = np.dot(self.params["W" + str(l + 1)], A) + self.params["b" + str(l + 1)]
            A = self.sigmoid(Z)
            store["A" + str(l + 1)] = A
            store["W" + str(l + 1)] = self.params["W" + str(l + 1)]
            store["Z" + str(l + 1)] = Z

        Z = np.dot(self.params["W" + str(self.num_layers-1)], A) + self.params["b" + str(self.num_layers-1)]
        A = self.softmax(Z)
        store["A" + str(self.num_layers-1)] = A
        store["W" + str(self.num_layers-1)] = self.params["W" + str(self.num_layers-1)]
        store["Z" + str(self.num_layers-1)] = Z

        return A, store

    def backward_propagation(self, X, Y, store):
        m = X.shape[0]

        store["A0"] = X.T

        A = store["A" + str(self.num_layers-1)]
        dZ = A - Y.T

        dW = np.dot(dZ, store["A" + str(self.num_layers - 2)].T) / m
        db = np.sum(dZ, axis=1, keepdims=True) / m
        dAPrev = store["W" + str(self.num_layers-1)].T.dot(dZ)

        self.grads["dW" + str(self.num_layers-1)] = dW
        self.grads["db" + str(self.num_layers-1)] = db

        for l in range(self.num_layers - 2, 0, -1):
            dZ = dAPrev * self.sigmoid_derivative(store["Z" + str(l)])
            dW = 1. / m * np.dot(dZ, store["A" + str(l - 1)].T)
            db = 1. / m * np.sum(dZ, axis=1, keepdims=True)
            if l > 1:
                dAPrev = store["W" + str(l)].T.dot(dZ)

            self.grads["dW" + str(l)] = dW
            self.grads["db" + str(l)] = db

    def update_params(self):
        for l in range(1, self.num_layers):
            self.params["W" + str(l)] = self.params["W" + str(l)] - self.learning_rate * self.grads[
                "dW" + str(l)]
            self.params["b" + str(l)] = self.params["b" + str(l)] - self.learning_rate * self.grads[
                "db" + str(l)]

    def fit(self, X_train, Y_train, X_val, Y_val):
        costs_train = []
        costs_val = []

        # Initialize parameters
        self.initialize_parameters()

        for i in range(self.num_epochs):
            cost_t = 0
            cost_v = 0

            mini_batches = self.create_minibatches(X_train, Y_train)

            for mini_batch in mini_batches:
                X_mini, Y_mini = mini_batch
                # Forward propagation
                A, _ = self.forward_propagation(X_mini)
                # Compute cost and add to the cost total
                cost_t += self.cross_entropy(Y_mini, A) / len(mini_batches)

            # Prediction for validation set
            A_val, store = self.forward_propagation(X_val)
            cost_v = self.cross_entropy(Y_val, A_val)

            for mini_batch in mini_batches:
                X_mini, Y_mini = mini_batch

                # Forward propagation
                _, store = self.forward_propagation(X_mini)
                # Backward propagation
                self.backward_propagation(X_mini, Y_mini, store)
                # Update parameters
                self.update_params()

            # Print cost after every epoch 10
            if i % 10 == 0:
                print("Epoch %i: Train cost %f, Valid cost %f" % (i, cost_t, cost_v))
            costs_train.append(cost_t)
            costs_val.append(cost_v)

        # plot the cost
        plt.plot(costs_train)
        plt.plot(costs_val)
        plt.ylabel('cost')
        plt.xlabel('epochs')
        plt.title(f"Learning rate={self.learning_rate}, epochs={self.num_epochs}, layers_dims={self.layers_dims[1:-1]}")
        plt.grid()
        plt.savefig(f"results\\graphs\\graph-{self.learning_rate},{self.num_epochs},layers_dims={self.layers_dims[1:-1]}.png")
        plt.clf()

    def evaluate(self, X, Y):
        # evaluating network accuracy
        A, cache = self.forward_propagation(X)
        y_hat = np.argmax(A, axis=0)
        y = np.argmax(Y, axis=1)
        accuracy = (y_hat == y).mean()
        return accuracy * 100


def load_dataset():
    (train_x, train_y), (val_x, val_y) = mnist.load_data()
    # Normalize
    train_x = train_x.reshape(train_x.shape[0], -1) / 255.
    train_x, test_x, train_y, test_y = train_test_split(train_x, train_y, test_size=1/6)
    val_x = val_x.reshape(val_x.shape[0], -1) / 255.

    enc = OneHotEncoder(sparse=False, categories='auto')
    train_y = enc.fit_transform(train_y.reshape(len(train_y), -1))
    val_y = enc.transform(val_y.reshape(len(val_y), -1))
    test_y = enc.transform(test_y.reshape(len(test_y), -1))

    return train_x, train_y, val_x, val_y, test_x, test_y


def create_files():
    Path("results\\graphs").mkdir(parents=True, exist_ok=True)
    Path("results\\data").mkdir(parents=True, exist_ok=True)

    f = open('results\\data\\data.csv', 'w')
    np.savetxt(f,
               [["hidden_layer", "learning_rate", "epochs", "learning_time", "acc_train", "acc_val", "acc_test"]],
               delimiter="; ",
               fmt='% s', )
    f.close()


def test_run():
    create_files()
    X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()  # loading and prepering data
    output_dim = 10
    hidden_layers_sizes = [10, 25, 50]
    learning_rates = [0.01, 0.05, 0.1, 0.5]
    epochs_array = [20, 50, 100, 200]

    hidden_layers_arr = []
    for L in range(1, len(hidden_layers_sizes) + 1):
        for subset in itertools.combinations(hidden_layers_sizes, L):
            for roll in itertools.product(subset, repeat=L):
                if list(roll) not in hidden_layers_arr:
                    hidden_layers_arr.append(list(roll))
    print(hidden_layers_arr)

    for epochs in epochs_array:
        for learning_rate in learning_rates:
            for hidden_layers in hidden_layers_arr:
                layers_dims = [X_train.shape[1], *hidden_layers, output_dim]  # defining layers dimensions

                model = Model(layers_dims, learning_rate=learning_rate, epochs=epochs, batch_size=32)  # learning session
                t = time.time()
                model.fit(X_train, y_train, X_val, y_val)
                elapsed = str(time.time() - t)
                print('Learning time: ' + elapsed)

                acc_tr = model.evaluate(X_train, y_train)  # prediction  and checking accuaracy of training set
                print("Accuracy for training set %f" % acc_tr)
                acc_val = model.evaluate(X_val, y_val)  # prediction  and checking accuaracy of validation set
                print("Accuracy for validation set %f" % acc_val)  # COS ZJEBANE
                acc_test = model.evaluate(X_test, y_test)  # prediction  and checking accuaracy of testset
                print("Accuracy for test set %f" % acc_test)

                f = open('results\\data\\data.csv', 'a')
                np.savetxt(f, [[hidden_layers, learning_rate, epochs, elapsed, acc_tr, acc_val, acc_test]],
                           delimiter="; ", fmt='% s', )
                f.close()


if __name__ == '__main__':
    test_run()
