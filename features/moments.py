import pandas as pd
from features.utils import SKIP_COLUMNS

def moments(question_dfs):
    question_moments = []

    for q in question_dfs:
        q_mean = q.iloc[:, SKIP_COLUMNS:].mean()
        q_mean.index = [col + '_mean' for col in q_mean.index]

        q_var = q.iloc[:, SKIP_COLUMNS:].var()
        q_var.index = [col + '_var' for col in q_var.index]

        q_skew = q.iloc[:, SKIP_COLUMNS:].skew()
        q_skew.index = [col + '_skew' for col in q_skew.index]

        q_kurt = q.iloc[:, SKIP_COLUMNS:].kurt()
        q_kurt.index = [col + '_kurt' for col in q_kurt.index]

        question_moments.append(pd.concat([q.iloc[0, :SKIP_COLUMNS], q_mean, q_var, q_skew, q_kurt], axis=0))

    return pd.concat(question_moments, axis=1).T
