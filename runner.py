import argparse
import pandas as pd
import os.path as path
import features


def extract_tsfresh_features():
    os.chdir(r'\users\gregory\dropbox\code\3deception')
    df = pd.read_csv('output_22-4-17_17-00-00.csv')
    from enum import IntEnum
    class RecordFlags(IntEnum):
        RECORD_FLAG_EMPTY = 0
        RECORD_FLAG_PAUSE = 1
        RECORD_FLAG_QUESTION = 2
        RECORD_FLAG_ANSWER_TRUE = 3
        RECORD_FLAG_ANSWER_FALSE = 4
        RECORD_FLAG_CONTROL_SHAPE_TRUE = 5
        RECORD_FLAG_CONTROL_SHAPE_FALSE = 6
        RECORD_FLAG_START_SESSION = 7
        RECORD_FLAG_END_SESSION = 8
        RECORD_FLAG_CHANGE = 9
        RECORD_FLAG_EXAMPLE_QUESTION = 10
        RECORD_FLAG_EXAMPLE_ANSWER_TRUE = 11
        RECORD_FLAG_EXAMPLE_ANSWER_FALSE = 12
        RECORD_FLAG_CATCH_ITEM = 13
    DROP_COLUMNS = ['question', 'record_flag', 'record_index']
    ANSWER_FLAGS = [RecordFlags.RECORD_FLAG_ANSWER_TRUE, RecordFlags.RECORD_FLAG_ANSWER_FALSE]
    SKIP_COLUMNS = len(DROP_COLUMNS)
    answers_df = df[df.record_flag.astype(int).isin(ANSWER_FLAGS)]
    qdfs = [t[1] for t in answers_df[answers_df.record_index != 2].groupby(DROP_COLUMNS)]
    from tsfresh import extract_features
    fdfs = []
    for d in qdfs:
        fdfs.append(d.drop(['record_flag', 'record_index'], axis=1))
    ddfs, y = [], []
    for i, x in enumerate(fdfs):
        ddfs.append(extract_features(x, column_id="question", column_sort="timestamp"))
        y.append(qdfs[i].record_flag.iloc[0])



def extract_features(raw_path, with_pca):
    features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Extracting features:")
    all_features = features.get_all_features(raw_df)

    print("Saving all features to {}...".format(features_path), end="")
    all_features.to_csv(features_path)

    if with_pca is not None:
        pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))
        print("Running PCA...")
        pca_features = features.utils.pca_3d(all_features, with_pca)
        print("Saving PCA features to {}...".format(pca_path), end="")
        pca_features.to_csv(pca_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='raw_path')
    parser.add_argument('-p', '--pca', dest='pca', type=int, default=None)

    args = parser.parse_args()

    if args.raw_path is None or not path.isfile(args.raw_path):
        print(r"Input csv doesn't exist: {}".format(args.raw_path))
        exit()

    extract_features(args.raw_path, args.pca)
