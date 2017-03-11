import pandas as pd
from scipy import stats as sp


def moments(question_dfs: list(pd.DataFrame)) -> list(pd.DataFrame):
    question_moments = []

    for q in question_dfs:
        question_moments.append(pd.concat([
            q.mean(),
            q.var(),
            q.aggregate(sp.skew, bias=False),
            q.aggregate(sp.kurtosis, bias=False)
        ], axis=1))

    return question_moments
