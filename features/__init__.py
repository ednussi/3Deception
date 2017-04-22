import pandas as pd
from . import utils, moments, discrete_states, dynamic, misc
from sklearn.decomposition import PCA


def PCA_that_vec(Panda_Data, Dim):

    # Turn data into np array without the ordering columns
    NP_Data = Panda_Data.iloc[:, 3:].values

    # Run PCA
    pca = PCA(n_components=Dim, copy=True, whiten=True)
    pca.fit(NP_Data)
    shrinked_data = pca.components_

    # Return to Pandas of proper size
    PCA_Panda_Data = pd.concat([Panda_Data.iloc[:, :3], pd.DataFrame(shrinked_data)],axis=1)

    return PCA_Panda_Data

def get_all_features(raw_df):
    print("Splitting answers... ", end="")
    question_idx_dfs = utils.split_df_to_questions(raw_df)
    print("Done.")

    print("Quantizing... ", end="")
    quantized_question_idx_dfs = utils.quantize(question_idx_dfs, n_quant=4)
    print("Done.")
    
    # print("Moments... ", end="")
    # all_moments = moments.moments(question_idx_dfs)
    # print("Done.")

    # print("Discrete... ", end="")
    # all_discrete = discrete_states.discrete_states(quantized_question_idx_dfs)
    # print("Done.")

    # print("Dynamic... ", end="")
    # all_dynamic = dynamic.dynamic(quantized_question_idx_dfs)
    # print("Done.")

    print("Miscellaneous... ", end="")
    all_misc = misc.misc(question_idx_dfs)
    print("Done.")

    all_features = pd.concat([
        # all_moments,
        # all_discrete.iloc[:, utils.SKIP_COLUMNS:],
        # all_dynamic,
        all_misc
    ], axis=1)

    return all_features