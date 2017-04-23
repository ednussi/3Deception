import argparse
import pandas as pd
import os.path as path
import features


def extract_features(raw_path, with_pca):
    features_path = "features_" + raw_path

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Extracting features:")
    all_features = features.get_all_features(raw_df)

    print("Saving all features to {}...".format(features_path), end="")
    all_features.to_csv(features_path)

    if with_pca is not None:
        print("Running PCA...")
        pca_features = features.utils.pca_3d(all_features, with_pca)
        print("Saving PCA features to pca_{}...".format(features_path), end="")
        pca_features.to_csv("pca_" + features_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='raw_path')
    parser.add_argument('-p', '--with_pca', dest='with_pca', type=int, default=None)

    args = parser.parse_args()

    if args.raw_path is None or not path.isfile(args.raw_path):
        print(r"Input csv doesn't exist: {}".format(args.raw_path))
        exit()

    extract_features(args.raw_path, args.with_pca)
