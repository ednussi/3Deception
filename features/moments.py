import pandas as pd
from scipy import stats as sp


def moments(df: pd.DataFrame) -> pd.DataFrame:
    df_mean = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).mean()
    df_var = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).var()
    df_skew = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).aggregate(sp.skew, bias=False)
    df_kurt = df.drop(['timestamp'], axis=1).groupby(['question', 'record_flag']).aggregate(sp.kurtosis, bias=False)

    return pd.concat([df_mean, df_var, df_skew, df_kurt], axis=1)
