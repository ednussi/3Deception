import pandas as pd
import numpy as np
from scipy import stats as sp
from sklearn import cluster as sk_cluster


def moments(df: pd.DataFrame) -> pd.DataFrame:
    df_mean = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).mean()
    df_var = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).var()
    df_skew = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).aggregate(sp.skew, bias=False)
    df_kurt = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).aggregate(sp.kurtosis, bias=False)

    return pd.concat([df_mean, df_var, df_skew, df_kurt], axis=1)


def quantize(df: pd.DataFrame, n_quant) -> pd.DataFrame:
    question_dfs = []
    for qdf in df.drop(['timestamp'], axis=1).groupby(['question']):
        for au in qdf[1].iloc[:, 2:]:
            qdf[1].loc[:, au] = sk_cluster.KMeans(n_clusters=n_quant, random_state=1).fit_predict(
                np.reshape(qdf[1][au].values, (-1, 1)))
        question_dfs.append(qdf)

    return question_dfs


def dynamic_activations(question_dfs: [pd.DataFrame]) -> pd.DataFrame:
    question_ratios = []
    question_levels = []
    question_length = []
    question_average_volumes = []

    # for each question df
    for qdf in question_dfs:
        # for each AU in the question

        # Activation ratio (#frames AU is nonzero / #frames)
        question_ratios.append((qdf[0], qdf[1].astype(bool).sum() / qdf[1].count()))

        # Activation mean level
        question_levels.append((qdf[0], qdf[1].mean()))

        # TODO Activation mean length

        # AU average volume
        question_average_volumes.append(pd.Series(qdf[1].astype(bool).sum().mean(axis=1), name='average_volume'))

    return question_ratios, question_levels, question_length, question_average_volumes
