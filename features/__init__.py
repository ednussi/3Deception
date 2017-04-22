import pandas as pd
from . import utils, moments, discrete_states, dynamic, misc
from sklearn.decomposition import PCA


def pca_3d(df, dim):

    # turn data into np array without the ordering columns
    NP_Data = df.iloc[:, 3:].values

    # run PCA
    pca = PCA(n_components=dim, copy=True, whiten=True)
    pca.fit(NP_Data)
    shrinked_data = pca.components_

    # return to DataFrame of proper size
    return pd.concat([df.iloc[:, :3], pd.DataFrame(shrinked_data)], axis=1)


def get_all_features(raw_df):
    print("Splitting answers... ", end="")
    question_idx_dfs = utils.split_df_to_questions(raw_df)
    print("Done.")

    print("Quantizing... ", end="")
    quantized_question_idx_dfs = utils.quantize(question_idx_dfs, n_quant=4)
    quantized_question_idx_dfs.to_csv('quantized.csv')
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

    return all_features