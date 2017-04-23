import argparse
import pandas as pd
import os.path as path
import features


def extract_features(raw_path, features_path, get_pca, with_pca):

    if get_pca:
        print("Reading {}...".format(raw_path))
        features_df = pd.read_csv(raw_path)

        print("Running PCA...")
        pca_features = features.pca_3d(features_df, 3)

        print("Saving PCA features to {}...".format(features_path), end="")
        pca_features.to_csv("pca_" + features_path)

    else:
        print("Reading {}...".format(raw_path))
        raw_df = pd.read_csv(raw_path)

        print("Extracting features:")
        all_features = features.get_all_features(raw_df)

        if with_pca:
            print("Running PCA...")
            pca_features = features.pca_3d(all_features, 3)
            print("Saving PCA features to {}...".format(features_path), end="")
            pca_features.to_csv("pca_" + features_path)

        print("Saving all features to {}...".format(features_path), end="")
        all_features.to_csv(features_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='raw_path')
    parser.add_argument('-o', '--output', dest='features_path')
    parser.add_argument('-p', '--pca', dest='get_pca', action='store_true')
    parser.add_argument('-P', '--with_pca', dest='with_pca', action='store_true')

    args = parser.parse_args()

    if args.raw_path is None or not path.isfile(args.raw_path):
        print(r"Input csv doesn't exist: {}".format(args.raw_path))
        exit()

    extract_features(args.raw_path, args.features_path, args.get_pca, args.with_pca)
