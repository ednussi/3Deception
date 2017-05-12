import argparse
import pandas as pd
import os.path as path
from features import utils



TODO
1. AU SELECTION HARD CODED USING DANIELs Utils.py
2. EXTRACT FEATURES BY GROUPS
3. PIPELINE
    AU SELECTION
    FEATURE SELECTION (Groups vs global, dimension (size))
    PCA (Groups vs global, dimension (size))
    LEARNER (Algorithm, Method (4v1 or seesion prediction), Hyper Parameters)
    REPORT



"run in shell command:"
"python3 runner.py -i output_22-4-17_17-00-00.csv"


def extract_features(raw_path, with_pca, choose_top_au_by_group):
    features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Choosing Top AU..")
    top_AU = utils.get_top_au(raw_df, method, num_au)

    if (choose_top_au_by_group):
        print("Extracting features for each group..")
        all_features = utils.get_all_features(top_AU)
        print("Choosing Top features from each group..")
        top_features = utils.take_top_features(all_features, 30)
    else:
        print("Extracting features for all..")
        all_features = utils.get_all_features(top_AU)
        print("Choosing Top features with best correlation")
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
    parser.add_argument('-au', '--au_top_by_group', dest='top_au', type=bool, default=True)
    parser.add_argument('-au_num', '--au_top_by_group', dest='top_au', type=bool, default=True)
    parser.add_argument('-au', '--au_top_by_group', dest='top_au', type=bool, default=True)

    args = parser.parse_args()

    if args.raw_path is None or not path.isfile(args.raw_path):
        print(r"Input csv doesn't exist: {}".format(args.raw_path))
        exit()

    extract_features(args.raw_path, args.pca,args.top_au)

"""
import os
os.chdir('C:/Users/owner/Desktop/Project/3deception/')
import argparse
import pandas as pd
import os.path as path
from features import utils
raw_df = pd.read_csv('new_csv.csv')
"""