import pandas as pd
import numpy as np
from scipy import stats as sp
from sklearn import cluster as skcluster


def moments(df: pd.DataFrame) -> pd.DataFrame:
    df_mean = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).mean()
    df_var = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).var()
    df_skew = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).aggregate(sp.skew, bias=False)
    df_kurt = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).aggregate(sp.kurtosis, bias=False)

    return pd.concat([df_mean, df_var, df_skew, df_kurt], axis=1)


def dynamic_quantized(df: pd.DataFrame, n_quant) -> pd.DataFrame:
    question_dfs = []
    for qdf in df.drop(['timestamp'], axis=1).groupby(['question']):
        question_df_aus = qdf[1].iloc[:, 2:]
        for au in question_df_aus:
            question_df_aus[au] = skcluster.KMeans(n_clusters=n_quant, random_state=1).fit_predict(
                np.reshape(question_df_aus[au].values, (-1, 1)))
        question_dfs.append(question_df_aus)

    return pd.concat(question_dfs, axis=0)
