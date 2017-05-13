from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
from . import utils


def transition_features(transition_matrix):
    s1 = [0, 1, 2]
    s2 = [1, 2, 3]
    slow_y = s1 + s2
    slow_x = s2 + s1

    slow_nominator = transition_matrix[slow_y, slow_x].sum()

    f1 = [0, 0, 1]
    f2 = [2, 3, 3]
    fast_y = f1 + f2
    fast_x = f2 + f1

    fast_nominator = transition_matrix[fast_y, fast_x].sum()

    denominator = transition_matrix.sum()

    return 1.*(slow_nominator+fast_nominator)/denominator, 1.*slow_nominator/denominator, 1.0*fast_nominator/denominator


def dynamic(question_quantized_dfs):
    """
    Calculate transition matrix of quantized AUs and count transitions between different clusters
    Args:
        question_quantized_dfs: list of quantized question data frames

    Returns:
        a data frame (shape=(#questions, #AUs*3) where index is question number,
            columns are AU_change_ratio,AU_slow_change_ratio, AU_fast_change_ratio for each au
    """
    all_transitions = {}

    for i, qdf in enumerate(question_quantized_dfs):
        q = qdf.iloc[:, utils.SKIP_COLUMNS:]  # don't work on audio_rms column

        all_transitions[i] = OrderedDict()

        for au in q:
            all_transitions[i][au] = np.zeros((4, 4))

            for (x, y), c in Counter(zip(q[au], q[au][1:])).items():
                all_transitions[i][au][x - 1, y - 1] += c  # TODO some warning about non-integer index

    # Concatenate all AU triples right, then all question down
    temp1 = []
    for k, q_transitions in all_transitions.items():
        temp2 = []

        for au, trans in q_transitions.items():
            trans_features = list(transition_features(trans))
            trans_index = [au + '_change_ratio', au + '_slow_change_ratio', au + '_fast_change_ratio']

            temp2.append(pd.DataFrame(trans_features, index=trans_index).T)
        temp1.append(pd.concat(temp2, axis=1))

    question_dynamic_features = pd.concat(temp1, axis=0)

    # Fix index
    question_dynamic_features.index = range(len(question_quantized_dfs))

    return question_dynamic_features
