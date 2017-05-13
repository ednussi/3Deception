import pandas as pd
from . import utils


def moments(question_dfs):
    question_moments = []

    for q in question_dfs:
        q_mean = q.iloc[:, utils.SKIP_COLUMNS:].mean()
        q_mean.index = [col + '_mean' for col in q_mean.index]

        q_var = q.iloc[:, utils.SKIP_COLUMNS:].var()
        q_var.index = [col + '_var' for col in q_var.index]

        q_skew = q.iloc[:, utils.SKIP_COLUMNS:].skew()
        q_skew.index = [col + '_skew' for col in q_skew.index]

        q_kurt = q.iloc[:, utils.SKIP_COLUMNS:].kurt()
        q_kurt.index = [col + '_kurt' for col in q_kurt.index]

        q_min = q.iloc[:, utils.SKIP_COLUMNS:].min()
        q_min.index = [col + '_min' for col in q_min.index]

        q_max = q.iloc[:, utils.SKIP_COLUMNS:].max()
        q_max.index = [col + '_max' for col in q_max.index]

        question_moments.append(pd.concat([q.iloc[0, :utils.SKIP_COLUMNS], q_mean, q_var, q_skew, q_kurt, q_min, q_max], axis=0))

    return pd.concat(question_moments, axis=1).T

"""
To get here for debug reasons:
os.chdir('/cs/engproj/3deception/3deception')
import features
from features import utils, moments, discrete_states, dynamic, misc
import pandas as pd
raw_df = pd.read_csv('output_22-4-17_17-00-00.csv')
question_dfs = utils.split_df_to_questions(raw_df)
q0 = question_dfs[0]
"""