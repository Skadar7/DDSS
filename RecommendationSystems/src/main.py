import numpy as np
import pandas as pd
import plotly.figure_factory as ff

l2_norm = lambda X: np.sqrt(np.sum(X ** 2, axis=1))


def cosine_similarity(X, Y=None):
    l2_x = l2_norm(X)
    if Y == None:
        Y = X
        l2_y = l2_x
    else:
        l2_y = l2_norm(Y)
    return (X @ Y.T) / (l2_x * l2_y)


def pearsonr(X):
    m = X.shape[1]
    res = np.eye(m)
    mean_ = X.mean(axis=1)
    for i in range(m - 1):
        for j in range(i + 1, m):
            t_i = (X[:, i] - mean_)
            t_j = (X[:, j] - mean_)
            denominator = np.sum(t_i ** 2) * np.sum(t_j ** 2)
            res[i, j] = np.sum(t_i * t_j) / denominator
            res[j, i] = res[i, j]

    return res


def jacquard(X, axis=0):
    m = X.shape[axis]
    n = X.shape[abs(int(axis - 1))]
    res = np.eye(m)
    for i in range(m - 1):
        for j in range(i + 1, m):
            res[i][j] = np.sum(X[:, i] == X[:, j]) / n
            res[j][i] = res[i][j]

    return res


if __name__ == '__main__':
    df = pd.read_csv('../data/movies_ratings.csv')
    df_dummies = pd.read_csv('../data/movies.csv')
    # columns_dict = dict(zip(range(len(df.columns)), df.columns))
    # print(columns_dict)
    # r = cosine_similarity(df.values)
    # print(r)
    r = jacquard(df_dummies.values, axis=0)
    print(r)
