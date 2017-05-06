import pandas as pd
from . import utils, moments, discrete_states, dynamic, misc


def get_all_features(raw_df):
    print("Splitting answers... ", end="")
    question_idx_dfs = utils.split_df_to_answers(raw_df)
    print("Done.")

    print("Quantizing... ", end="")
    quantized_question_idx_dfs = utils.quantize(question_idx_dfs, n_quant=4)
    print("Done.")

    print("Moments... ", end="")
    all_moments = moments.moments(question_idx_dfs)
    print("Done.")

    print("Discrete... ", end="")
    all_discrete = discrete_states.discrete_states(quantized_question_idx_dfs)
    print("Done.")

    print("Dynamic... ", end="")
    all_dynamic = dynamic.dynamic(quantized_question_idx_dfs)
    print("Done.")

    print("Miscellaneous... ", end="")
    all_misc = misc.misc(question_idx_dfs)
    print("Done.")

    all_features = pd.concat([
        all_moments,
        all_discrete.iloc[:, utils.SKIP_COLUMNS:],
        all_dynamic,
        all_misc
    ], axis=1)

    all_features.index = all_moments.index

    return all_features

def correlate_features(features_pd,top_features_num):
    # top_features_num - How many features u want
    # return pandas of name of feature and its correlation
    correlation_to_flag = abs(features_pd.corr()['record_flag'])
    correlation_to_flag.sort(ascending=False)
    correlation_to_flag = correlation_to_flag.drop('record_flag')
    top_features = correlation_to_flag[0:top_features_num]
    return top_features

def take_top_features(features_pd,top_features_num):
    top_features = correlate_features(features_pd, top_features_num)
    return features_pd[top_features.index]

"""
To get here for debug reasons:
import os
os.chdir('/cs/engproj/3deception/3deception')
import features
from features import utils, moments, discrete_states, dynamic, misc
import pandas as pd
raw_df = pd.read_csv('output_22-4-17_17-00-00.csv')
question_idx_dfs = utils.split_df_to_questions(raw_df)
"""