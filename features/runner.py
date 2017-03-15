import discrete_states, dynamic, moments, utils, miscl
import pandas as pd


def get_all_features(raw_df):
    question_idx_dfs = utils.split_df_to_questions(raw_df)

    quantized_question_idx_dfs = utils.quantize(question_idx_dfs, n_quant=4)

    all_moments = moments.moments(question_idx_dfs)
    all_discrete = discrete_states.discrete_states(quantized_question_idx_dfs)
    all_dynamic = dynamic.dynamic(quantized_question_idx_dfs)
    all_miscl = miscl.miscl(question_idx_dfs)
    # TODO Misc features

    all_features = pd.concat([
        all_moments,
        all_discrete.iloc[:, utils.SKIP_COLUMNS:],
        all_dynamic.iloc[:, utils.SKIP_COLUMNS:],
        all_miscl


    ],axis=1)

    return all_features

def main():
    save_path = 'all_features.csv'
    data_file_path = '/cs/usr/ednussi/safe/3deception/questionnaire/data/rotembenhamo92@gmail.com/output/fs_shapes.1485101576.3601818.csv'

    raw_df_from_csv = pd.read_csv(data_file_path)
    all_features = get_all_features(raw_df_from_csv)
    all_features.to_csv(save_path)


if __name__ == "__main__":
    main()