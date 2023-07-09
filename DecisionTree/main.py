import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn import datasets
import pandas as pd

class Node:
    def __init__(
            self,
            feature=None,
            threshold=None,
            left=None,
            right=None,
            *,
            value=None
    ):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None


class DecsisionTree:
    def __init__(
            self,
            min_samples_split=2,
            max_depth=100,
            n_features=None
    ):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None

    def fit(self, X, y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)

        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check stopping criteria

        if depth >= self.max_depth or n_labels == 1 or n_samples <= self.min_samples_split:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find best split

        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes

        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)

        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)

        return Node(best_feature, best_thresh, left, right)

    def _most_common_label(self, y):
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        return value

    def _best_split(self, X, y, feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, threshold)
                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = threshold
        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):

        # parent entropy
        parent_entropy = self._entropy(y)
        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        n_l, n_r = len(left_idxs), len(right_idxs)

        if n_l == 0 or n_r == 0:
            return 0
        # calculate weighted avg. entropy of children
        n = len(y)
        E_l, E_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * E_l + (n_r / n) * E_r
        IG = parent_entropy - child_entropy
        return IG

        # calculate information gain (IG)

    def _split(self, X_column, threshold):
        left_idxs = np.argwhere(X_column <= threshold).flatten()
        right_idxs = np.argwhere(X_column > threshold).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        hist = np.bincount(y)
        ps = hist / len(y)  # p(xi)/n for each xi
        return -np.sum(p * np.log(p) for p in ps if p > 0)

    def predict(self, X):
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)


def accuracy(preds, targets):
    return np.sum(preds == targets) / len(targets)


if __name__ == '__main__':
    dataset = datasets.load_iris()
    X, y = dataset.data, dataset.target
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1243)
    decision_tree = DecsisionTree()
    decision_tree.fit(X_train, y_train)
    y_pred = decision_tree.predict(X_test)

    res_df = pd.DataFrame(
        data=np.c_[X_test, y_test, y_pred],
        columns=dataset.feature_names + ['target', 'prediction']
    )
    res_df.target = res_df.target.astype(int)
    res_df.prediction = res_df.prediction.astype(int)
    print(f"Accuracy: {accuracy(y_test, y_pred)}")
    print(res_df)
