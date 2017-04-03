import pandas as pd
import numpy as np
from . import utils


def discrete_states(question_quantized_dfs):
    question_ratios = []
    question_levels = []
    question_lengths = []
    question_average_volumes = []

    cols = question_quantized_dfs[0].iloc[:, utils.SKIP_COLUMNS:].columns

    # for each question df
    for qdf in question_quantized_dfs:
        q = qdf.iloc[:, utils.SKIP_COLUMNS:]

        # for each AU in the question

        nonzero = q.astype(bool).sum()

        # Activation ratio (#frames AU is nonzero / #frames)
        activation_ratios = utils.scale(nonzero / q.count())
        activation_ratios.index = [col + '_ratio' for col in cols]  # Update index
        activation_ratios = pd.concat([q.iloc[0, :utils.SKIP_COLUMNS], activation_ratios], axis=0)  # Plug metadata
        question_ratios.append(activation_ratios)

        # Activation mean level
        activation_mean_level = utils.scale(q.mean())
        activation_mean_level.index = [col + '_avg_level' for col in cols]
        question_levels.append(activation_mean_level)

        # Activation mean length
        lengths = []
        for col in q:
            lengths.append(np.array(utils.runs_of_ones_list(q[col] > 1)).mean())
        activation_mean_length = pd.Series(lengths, index=[col + '_avg_length' for col in cols])
        question_lengths.append(activation_mean_length)

        # AU average volume
        question_average_volumes.append(nonzero.mean())

    question_average_volumes = pd.DataFrame(question_average_volumes, columns=['avg_volume']).fillna(0.0)

    return pd.concat(
        [pd.concat(question_ratios, axis=1).T.fillna(0.0),
         pd.concat(question_levels, axis=1).T.fillna(0.0),
         pd.concat(question_lengths, axis=1).T.fillna(0.0),
         question_average_volumes],
        axis=1)
