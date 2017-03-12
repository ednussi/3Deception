from collections import Counter, OrderedDict
import numpy as np
import pandas as pd
from features.utils import SKIP_COLUMNS


def transition_features(transition_matrix):
    side = transition_matrix.shape[0]
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
        question_dfs: list of quantized question data frames

    Returns:
        a data frame (shape=(#questions, #AUs*3) where index is question number,
            columns are AU_change_ratio,AU_slow_change_ratio, AU_fast_change_ratio for each au
    """
    all_transitions = {}

    for qdf in question_quantized_dfs:
        q = qdf.iloc[:, SKIP_COLUMNS:]

        skipped = qdf.iloc[:, :SKIP_COLUMNS]
        # key is 2D tuple of values of all skipped columns
        # e.g. (('question', 'record_flag', 'record_idx'), (1, 1, 1))
        all_transitions[(tuple(skipped.index), tuple(skipped.values))] = OrderedDict()

        for au in q:
            all_transitions[-1][au] = np.zeros((4, 4))

            for (x, y), c in Counter(zip(q[au], q[au][1:])).items():
                all_transitions[-1][au][x - 1, y - 1] += c  # TODO some warning about non-integer index

    # Concatenate all AU triples right, then all question down
    question_dynamic_features = pd.concat([
        pd.concat([
            pd.DataFrame(list(k[1]) + list(transition_features(trans)),
                         index=list(k[0]) + [au + '_change_ratio', au + '_slow_change_ratio', au + '_fast_change_ratio']).T
            for au, trans in q_transitions.items()], axis=1)
        for k, q_transitions in all_transitions.items()], axis=0)

    # Fix index
    question_dynamic_features.index = range(len(question_quantized_dfs))

    return question_dynamic_features