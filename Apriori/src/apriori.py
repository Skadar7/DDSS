import numpy as np
import pandas as pd
from itertools import combinations, chain


def subsets(arr):
    """ Returns non empty subsets of arr"""
    return list(chain(*[combinations(arr, i + 1) for i, a in enumerate(arr)]))


# join set with itself
def generate_new_combinations(old_combinations):
    items_types_in_previous_step = np.unique(old_combinations.flatten())
    for old_combination in old_combinations:
        max_combination = old_combination[-1]
        mask = items_types_in_previous_step > max_combination
        valid_items = items_types_in_previous_step[mask]
        old_tuple = tuple(old_combination)
        for item in valid_items:
            yield from old_tuple
            yield item


def apriori(df, min_support=0.5, use_colnames=True, max_len=None):
    def _support(_x, _n_rows):
        out = np.sum(_x, axis=0) / _n_rows
        return np.array(out).reshape(-1)

    if min_support <= 0.0:
        raise ValueError(
            "`min_support` must be a positive "
            "number within the interval `(0, 1]`. "
            "Got %s." % min_support
        )
    X = df.values
    support = _support(X, X.shape[0])
    ary_col_idx = np.arange(X.shape[1])
    support_dict = {1: support[support >= min_support]}
    itemset_dict = {1: ary_col_idx[support >= min_support].reshape(-1, 1)}
    max_itemset = 1
    rows_count = float(X.shape[0])

    all_ones = np.ones((int(rows_count), 1))

    while max_itemset and max_itemset < (max_len or float("inf")):
        next_max_itemset = max_itemset + 1

        combin = generate_new_combinations(itemset_dict[max_itemset])
        combin = np.fromiter(combin, dtype=int)
        combin = combin.reshape(-1, next_max_itemset)

        if combin.size == 0:
            break
        _bools = np.all(X[:, combin], axis=2)
        support = _support(np.array(_bools), rows_count)
        _mask = (support >= min_support).reshape(-1)
        if any(_mask):
            itemset_dict[next_max_itemset] = np.array(combin[_mask])
            support_dict[next_max_itemset] = np.array(support[_mask])
            max_itemset = next_max_itemset
        else:
            # Exit condition
            break

    all_res = []
    for k in sorted(itemset_dict):
        support = pd.Series(support_dict[k])
        itemsets = pd.Series([frozenset(i) for i in itemset_dict[k]], dtype="object")

        res = pd.concat((support, itemsets), axis=1)
        all_res.append(res)

    res_df = pd.concat(all_res)
    res_df.columns = ["support", "itemsets"]
    if use_colnames:
        mapping = {idx: item for idx, item in enumerate(df.columns)}
        res_df["itemsets"] = res_df["itemsets"].apply(
            lambda x: frozenset([mapping[i] for i in x])
        )
    res_df = res_df.reset_index(drop=True)

    return res_df


def association_rules(input_df, min_support, min_confidence):
    metric_dict = {
        "condition support": lambda _, support_condition, __: support_condition,
        "consequence support": lambda _, __, support_consequence: support_consequence,
        "support": lambda support, _, __: support,
        "confidence": lambda support, support_condition, _: support / support_condition,
        "lift": lambda support, support_condition, support_consequence:
        metric_dict["confidence"](support, support_condition, support_consequence) / support_consequence,
        "leverage": lambda support, support_condition, support_consequence:
        metric_dict["support"](support, support_condition, support_consequence) - support_condition * support_consequence
    }

    columns_ordered = [
        "condition support",
        "consequence support",
        "support",
        "confidence",
        "lift",
        "leverage"
    ]

    rule_antecedents = []
    rule_consequents = []
    rule_supports = []

    df = apriori(input_df, min_support=min_support)
    # create dict out of dataframe
    keys = df["itemsets"].values
    values = df["support"].values
    frozenset_vect = np.vectorize(lambda x: frozenset(x))
    frequent_items_dict = dict(zip(frozenset_vect(keys), values))

    for k in frequent_items_dict.keys():
        support = frequent_items_dict[k]
        all_subsets = subsets(k)
        for subset in all_subsets:
            condition = frozenset(subset)
            consequence = k.difference(condition)
            if len(consequence) == 0 or len(condition) == 0:
                continue
            try:
                support_condition = frequent_items_dict[condition]
                support_consequence = frequent_items_dict[consequence]
            except KeyError as e:
                raise KeyError(str(e))
            confidence = metric_dict["confidence"](support, support_condition, support_consequence)
            support = metric_dict["support"](support, support_condition, support_consequence)
            if confidence >= min_confidence and support >= min_support:
                rule_antecedents.append(condition)
                rule_consequents.append(consequence)
                rule_supports.append([support, support_condition, support_consequence])

    if not rule_supports:
        return pd.DataFrame(columns=["condition", "consequence"] + columns_ordered)

    else:
        # generate metrics
        rule_supports = np.array(rule_supports).T.astype(float)
        df_res = pd.DataFrame(
            data=list(zip(rule_antecedents, rule_consequents)),
            columns=["condition", "consequence"],
        )
        support = rule_supports[0]
        support_condition = rule_supports[1]
        support_consequence = rule_supports[2]
        for m in columns_ordered:
            df_res[m] = metric_dict[m](support, support_condition, support_consequence)

        return df_res


if __name__ == '__main__':
    df = pd.read_csv('../data/movies.csv')
    res_df = association_rules(df, min_support=0.07, min_confidence=0.6)
