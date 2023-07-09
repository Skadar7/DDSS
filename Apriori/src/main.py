from itertools import chain
import warnings
from mlxtend.frequent_patterns import apriori
import pandas as pd

warnings.filterwarnings("ignore")


def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    pool = list(iterable)
    n = len(pool)
    if r > n:
        return
    indices = list(range(r))
    yield list(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i + 1, r):
            indices[j] = indices[j - 1] + 1
        yield list(pool[i] for i in indices)


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)])


def join_set(item_set, length):
    """Join a set with itself and returns the n-element itemsets"""
    return list(
        [list(set(i).union(set(j))) for i in item_set for j in item_set if len(set(i).union(set(j))) == length]
    )


class Apriori:
    def __init__(self, min_support, df):
        self.df = df
        self.initial_set = self._init_set(df)
        self.min_support = min_support

    def _init_set(self, df):
        t = df.columns.to_list()[1:]
        t = list(map(lambda x: [x], t))
        return t

    def get_items_with_min_support(self, current_set):
        to_delete = []

        def check_item(item):
            t = self.df[item].sum(axis=1)
            t = t[t == len(item)]
            return t

        def calculate_support(item):
            t = check_item(item)
            support = len(t) / len(self.df)
            return support

        for item in current_set:
            if len(item) > 1:
                item_subsets = list(combinations(item, len(item) - 1))
            else:
                support = calculate_support(item)
                if support < self.min_support:
                    to_delete.append(item)
                continue
            for item_subset in item_subsets:
                support = calculate_support(item_subset)
                if support < self.min_support:
                    to_delete.append(item)
                    break
        current_set = [x for x in current_set if x not in to_delete]
        return current_set

    def run_apriori(self):
        current_set = self.get_items_with_min_support(self.initial_set)
        large_set = {}
        k = 2
        while len(current_set) > 0:
            print(f'CURRENT SET\n{len(current_set)}')
            large_set[k - 1] = current_set[:]
            current_set = join_set(current_set, k)
            current_set = self.get_items_with_min_support(current_set)
            k += 1
        print(large_set[k - 2])


if __name__ == '__main__':
    df = pd.read_csv('../data/movies.csv')
    app = Apriori(0.07, df)
    app.run_apriori()
