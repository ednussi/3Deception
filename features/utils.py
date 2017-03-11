from sklearn import cluster as sk_cluster
import pandas as pd
import numpy as np


def split_df_to_questions(df: pd.DataFrame) -> list(pd.DataFrame):
    """
    Split raw data frame by questions
    """
    return [t[1] for t in df.drop(['timestamp'], axis=1).groupby(['question'])]


def scale(val):
    """
    Scale the given value from the scale of val to the scale of bot-top.
    """
    bot = 0
    top = 1

    if max(val)-min(val) == 0:
        return val
    return ((val - min(val)) / (max(val)-min(val))) * (top-bot) + bot


def quantize(question_dfs: list(pd.DataFrame), n_quant: int) -> list(pd.DataFrame):
    question_quantized_dfs = []

    for q_df in question_dfs:
        q = q_df.copy()
        for au in q.iloc[:, 2:]:
            q.loc[:, au] = sk_cluster.KMeans(n_clusters=n_quant, random_state=1)\
                .fit_predict(np.reshape(q[au].values, (-1, 1)))

        question_quantized_dfs.append(q)

    return question_quantized_dfs
