import pandas as pd
from features import utils as utils


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
