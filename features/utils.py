import itertools
from sklearn import cluster as sk_cluster
import numpy as np


DROP_COLUMNS = ['question', 'record_flag', 'record_idx']
SKIP_COLUMNS = len(DROP_COLUMNS)


def split_df_to_questions(df):
    """
    Split raw data frame by questions
    """
    return [t[1] for t in df.drop(['timestamp'], axis=1).groupby(DROP_COLUMNS)]


def scale(val):
    """
    Scale the given value from the scale of val to the scale of bot-top.
    """
    bot = 0
    top = 1

    if max(val)-min(val) == 0:
        return val
    return ((val - min(val)) / (max(val)-min(val))) * (top-bot) + bot


def quantize(question_dfs, n_quant):
    """
    Quantize values in data frames using K-means
    Args:
        question_dfs: data frames, note that first two columns are not quantized
        n_quant: number of clusters

    Returns:
        list of quantized data frames
    """
    question_quantized_dfs = []

    for q_df in question_dfs:
        q = q_df.copy()
        for au in q.iloc[:, SKIP_COLUMNS:]:
            q.loc[:, au] = sk_cluster.KMeans(n_clusters=n_quant, random_state=1)\
                .fit_predict(np.reshape(q[au].values, (-1, 1)))

        question_quantized_dfs.append(q)

    return question_quantized_dfs


def runs_of_ones_list(bits):
    """
    Calculate lengths of continuous sequences of ones in binary iterable
    Args:
        bits: binary iterable

    Returns:
        array of lengths
    """
    return [sum(g) for b, g in itertools.groupby(bits) if b]