import pandas as pd


def moments(question_dfs):
    question_moments = []

    for q in question_dfs:
        question_moments.append(pd.concat([q.mean(), q.var(), q.skew(), q.kurt()], axis=1))

    return question_moments
