import pandas as pd
import numpy as np
from features import utils as utils


def discrete_states(question_quantized_dfs):
    question_ratios = []
    question_levels = []
    question_lengths = []
    question_average_volumes = []

    cols = question_quantized_dfs[0].columns

    # for each question df
    for q in question_quantized_dfs:
        # for each AU in the question

        nonzero = q.astype(bool).sum()

        # Activation ratio (#frames AU is nonzero / #frames)
        question_ratios.append(utils.scale(nonzero / q.count()))
        question_ratios[-1].index = [col + '_ratio' for col in cols]

        # Activation mean level
        question_levels.append(utils.scale(q.mean()))
        question_levels[-1].index = [col + '_avg_level' for col in cols]

        # Activation mean length
        lengths = []
        for col in q:
            lengths.append(np.array(utils.runs_of_ones_list(q[col] > 1)).mean())
        question_lengths.append(pd.Series(lengths, index=[col + '_avg_length' for col in cols]))

        # AU average volume
        question_average_volumes.append(nonzero.mean())

    question_average_volumes = pd.DataFrame(question_average_volumes, columns=['avg_volume'])\
        .fillna(0.0)

    return pd.concat(
        [pd.concat(question_ratios, axis=1).T.fillna(0.0),
         pd.concat(question_levels, axis=1).T.fillna(0.0),
         pd.concat(question_lengths, axis=1).T.fillna(0.0),
         question_average_volumes],
        axis=1)
