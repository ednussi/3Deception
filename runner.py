import argparse
import pandas as pd
import os.path as path
from features import utils


"run in shell command:"
"python3 runner.py -i output_22-4-17_17-00-00.csv"


def extract_features(raw_path, with_pca):
    features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Extracting features:")
    all_features = utils.get_all_features(raw_df)

    print("Choosing features with best correlation")
    top_features = utils.take_top_features(all_features,30)

    print("Saving all features to {}...".format(features_path), end="")
    top_features.to_csv(features_path)

    if with_pca is not None:
        pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))
        print("Running PCA...")
        pca_features = utils.pca_3d(top_features, with_pca)
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
