from collections import Counter
import numpy as np


def dynamic(question_quantized_dfs):
    """
    Calculate transition matrix of quantized AUs and count transitions between different clusters
    Args:
        question_dfs: list of quantized question data frames

    Returns:
        a data frame (shape=(#questions, #AUs*3) where index is question number,
            columns are AU_change_ratio,AU_slow_change_ratio, AU_fast_change_ratio for each au
    """
    all_transitions = []
    for q in question_quantized_dfs:
        all_transitions.append({})

        for au in q:
            all_transitions[-1][au] = np.zeros((4, 4))

            for (x, y), c in Counter(zip(q[au], q[au][1:])).items():
                all_transitions[-1][au][x - 1, y - 1] += c


    pass