import argparse
import pandas as pd
import os.path as path
import features


def main(raw_path, features_path):
    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)
    print("Extracting features:")
    all_features = features.get_all_features(raw_df)
    print("Saving to {}...".format(features_path), end="")
    all_features.to_csv(features_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='raw_path')
    parser.add_argument('-o', '--output', dest='features_path')

    args = parser.parse_args()

    if args.raw_path is None or not path.isfile(args.raw_path):
        print(r"Raw data csv doesn't exist: {}".format(args.raw_path))
        exit()

    main(args.raw_path, args.features_path)
