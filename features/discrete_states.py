import pandas as pd
import numpy as np
from sklearn import cluster as sk_cluster
import features.utils as utils


def quantize(question_dfs: list(pd.DataFrame), n_quant: int) -> list(pd.DataFrame):
    question_quantized_dfs = []

    for q_df in question_dfs:
        q = q_df.copy()
        for au in q.iloc[:, 2:]:
            q.loc[:, au] = sk_cluster.KMeans(n_clusters=n_quant, random_state=1)\
                .fit_predict(np.reshape(q[au].values, (-1, 1)))

        question_quantized_dfs.append(q)

    return question_quantized_dfs


def discrete_states(question_quantized_dfs: list(pd.DataFrame)) -> list(pd.DataFrame):
    question_ratios = []
    question_levels = []
    question_length = []
    question_average_volumes = []

    # for each question df
    for q in question_quantized_dfs:
        # for each AU in the question

        # Activation ratio (#frames AU is nonzero / #frames)
        question_ratios.append((q.astype(bool).sum() / q.count()).apply(utils.scale))

        # Activation mean level
        question_levels.append(q.mean().apply(utils.scale))

        # TODO Activation mean length

        # AU average volume
        question_average_volumes.append(
            pd.Series(q.astype(bool).sum().mean(axis=1), name='average_volume'))

    return question_ratios, question_levels, question_length, question_average_volumes
