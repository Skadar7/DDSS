import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def encode_ratings(x):
    if x <= 0:
        return 0
    if x >= 1:
        return 1


ratings_df = pd.read_csv('../data/ratings_small.csv')

movies_df = pd.read_csv('../data/movies_metadata.csv')

title_mask = movies_df['title'].isna()
movies_df = movies_df.loc[title_mask == False]

movies_df = movies_df.astype({'id': 'int64'})

df = pd.merge(ratings_df, movies_df[['id', 'title']], left_on='movieId', right_on='id')

df.drop(['id', 'timestamp'], axis=1, inplace=True)

df = df.drop_duplicates(['userId', 'title'])

df_pivot = df.pivot(index='userId', columns='title', values='rating').fillna(0)

df_pivot = df_pivot.astype('int64')

df_pivot = df_pivot.applymap(encode_ratings)

df_pivot.to_csv('../data/movies.csv', index=False)