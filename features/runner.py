from features import discrete_states, dynamic, moments, utils
import pandas as pd


def get_all_features(raw_df):
    question_idx_dfs = utils.split_df_to_questions(raw_df)

    quantized_question_idx_dfs = utils.quantize(question_idx_dfs, n_quant=4)

    all_moments = moments.moments(question_idx_dfs)
    all_discrete = discrete_states.discrete_states(quantized_question_idx_dfs)
    all_dynamic = dynamic.dynamic(quantized_question_idx_dfs)
    # TODO Misc features

    all_features = pd.concat([
        all_moments,
        all_discrete.iloc[:, utils.SKIP_COLUMNS:],
        all_dynamic.iloc[:, utils.SKIP_COLUMNS:]
    ])

    return all_features