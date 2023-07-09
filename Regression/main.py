import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from sklearn import datasets
from sklearn.model_selection import train_test_split
from abc import ABC, abstractmethod
from tqdm import tqdm
import pandas as pd

matplotlib.use('Qt5Agg')


def one_hot(y, num_classes):
    y_full = np.zeros((len(y), num_classes)).astype(int)
    for j, yj in enumerate(y):
        y_full[j, yj] = 1
    return y_full


class Regression(ABC):
    def __init__(self):
        self.W = None
        self.b = None
        self.batch_idx = 0
        self.barch_size = None

    @abstractmethod
    def init_weights(self, num_features):
        pass

    def parameters(self):
        return [self.W, self.b]

    def forward(self, X):
        return X @ self.W + self.b

    @abstractmethod
    def loss(self, outputs, targets):
        pass

    def calculate_gradients(self, batch_x, outputs, batch_y):
        batch_size = len(batch_x)
        if isinstance(self, LogisticRegression):
            batch_y = one_hot(batch_y, self._num_classes)
        dW = (2 / batch_size) * batch_x.T @ (outputs - batch_y)
        db = (2 / batch_size) * np.sum(outputs - batch_y)
        return dW, db

    def fit(self, X, y, epochs, batch_size, lr, verbose=True):
        num_train = X.shape[0]
        num_features = X.shape[1]
        self._num_classes = np.max(y) + 1

        self.init_weights(num_features)

        history = []
        if verbose:
            loop = tqdm(range(epochs))
        else:
            loop = range(epochs)
        for epoch in loop:
            running_loss = 0
            shuffled_indices = np.arange(num_train)
            np.random.shuffle(shuffled_indices)
            sections = np.arange(batch_size, num_train, batch_size)
            batches_indices = np.array_split(shuffled_indices, sections)
            batch_number = np.random.randint(len(batches_indices))
            batch_x = X[batches_indices[batch_number]]
            batch_y = y[batches_indices[batch_number]]

            outputs = self.forward(batch_x)
            running_loss = self.loss(outputs, batch_y)
            dW, db = self.calculate_gradients(batch_x, outputs, batch_y)
            self.W = self.W - lr * dW
            self.b = self.b - lr * db

            if epoch % 100 == 0:
                history.append(running_loss)
        return history

    def predict(self, X):
        return self.forward(X)


class LinearRegression(Regression):
    def loss(self, outputs, labels):
        return np.mean((outputs - labels) ** 2)

    def init_weights(self, num_features):
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features)
        if self.b is None:
            self.b = 0.001 * np.random.randn(1)


class LogisticRegression(Regression):
    def softmax(self, preds):
        # for N shape predictions
        exps = np.exp(preds)
        if preds.shape == (len(preds),):
            return exps / np.sum(exps)
        else:
            return exps / np.sum(exps, axis=1, keepdims=True)

    def init_weights(self, num_features):
        if self.W is None:
            self.W = 0.001 * np.random.randn(num_features, self._num_classes)
        if self.b is None:
            self.b = 0.001 * np.random.randn(self._num_classes)

    def forward(self, X):
        preds = super().forward(X)
        return self.softmax(preds)

    def loss(self, outputs, labels):
        if outputs.shape == (len(outputs),):
            return -np.log(outputs[0, labels])
        else:
            # сумма ошибок на различных элементах батча
            return np.sum(-np.log(np.array([outputs[j, labels[j]] for j in range(len(labels))])))

    def predict(self, X):
        probabilities = super().predict(X)
        return np.argmax(probabilities, axis=1)


def accuracy(preds, targets):
    return np.sum(preds == targets) / len(targets)


def calculate_metrics(x, y, preds):
    n = len(x)
    Q = sum((i - np.mean(y)) ** 2 for i in y)
    QR = sum((i - np.mean(y)) ** 2 for i in preds)
    r_determ = QR / Q
    a = sum(i * j for i, j in zip(x, y))
    b = sum(x) * sum(y) / n
    c = (sum(i ** 2 for i in x) - (sum(x) ** 2) / n) ** 0.5
    d = (sum(i ** 2 for i in y) - (sum(y) ** 2) / n) ** 0.5
    r_corr = ((a - b) / (c * d)) ** 0.5
    return r_determ, r_corr[0]


if __name__ == '__main__':
    # x = [70, 56, 29, 26, 65, 94, 71, 28, 75, 78]
    # y = [5,
    #      2.5,
    #      1.5,
    #      1.3,
    #      2.8,
    #      10,
    #      6,
    #      1.2,
    #      6,
    #      6.1]
    # print(sum([i ** 2 for i in x]))
    # a = sum([i * j for i, j in zip(x, y)])
    # b = sum(x) * sum(y)
    # c = sum([i ** 2 for i in x])
    # d = sum(x) ** 2
    # print(a, b, c, d)
    # b1 = (a - b) / (c - d)
    # print(np.mean(y), np.mean(x))
    # b0 = np.mean(y) - b1 * np.mean(x)
    # print(b0, b1)
    # preds = list(map(lambda i: 0.038 + 1.56 + 0.0709 * i, x))
    # r_determ, r_corr = calculate_metrics(x, y, preds)
    # print(r_corr)
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.set_xlabel('sapce, m^2')
    # ax.set_ylabel('price, m.')
    # plt.plot(x, preds, color='black')
    # #
    # plt.plot(x, y, 'o')
    # plt.show()

    # X, y = datasets.make_regression(n_samples=100, n_features=1, noise=20, random_state=4)
    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1243)
    # linreg = LinearRegression()
    # history = linreg.fit(X, y, epochs=1000, batch_size=10, lr=1e-3)
    # preds = linreg.predict(X_test)
    # r_determ, r_corr = calculate_metrics(X_test, y_test, preds)
    # print(f'Коэффициент детерминации: {r_determ}\nКоэффициент корреляции: {r_corr}')
    # fig, axes = plt.subplots(1, 2, figsize=(16, 12))
    # axes[0].scatter(X_test, y_test, s=10)
    # axes[0].plot(X_test, preds, color='black', linewidth=2)
    # axes[0].set_title('Test result')
    # axes[1].plot(history)
    # axes[1].set_title('MSE loss')
    # plt.show()

    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1243)
    logreg = LogisticRegression()
    history = logreg.fit(X_train, y_train, epochs=4500, batch_size=20, lr=1e-3)
    y_pred = logreg.predict(X_test)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_title('Loss')
    ax.set_ylabel('CEL')
    ax.set_xlabel('Epoch')
    res_df = pd.DataFrame(
        data=np.c_[X_test, y_test, y_pred],
        columns=dataset.feature_names + ['target', 'prediction']
    )
    res_df.target = res_df.target.astype(int)
    res_df.prediction = res_df.prediction.astype(int)
    print(f"Accuracy: {accuracy(y_test, y_pred)}")
    print(res_df)
    ax.plot(history)
    plt.show()
