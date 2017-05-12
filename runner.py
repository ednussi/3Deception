import argparse
import pandas as pd
import os.path as path
from features import utils



TODO
2. EXTRACT FEATURES BY GROUPS
3. PIPELINE
    AU SELECTION
    FEATURE SELECTION (Groups vs global, dimension (size))
    PCA (Groups vs global, dimension (size))
    LEARNER (Algorithm, Method (4v1 or seesion prediction), Hyper Parameters) - gregory
    REPORT



"run in shell command:"
"python3 runner.py -i output_22-4-17_17-00-00.csv"


def extract_features(raw_path, with_pca, au, au_num, feat, feat_num):
    features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Choosing Top AU with method",au)
    top_AU = utils.get_top_au(raw_df, au, au_num)

    print("Extracting features with method:",feat)
    top_features = utils.get_top_features(top_AU, feat,feat_num)

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

    # Possible Choices: 'daniel' 'mouth' 'eyes' 'brows' 'eyes_area' 'smile' 'blinks' 'top'
    parser.add_argument('-au', '--au_choice', dest='au', type=str, default=None)
    # If au was chosen 'top' need a number for number of top AU
    parser.add_argument('-au_num', '--au_top_num', dest='au_num', type=int, default=24)

    # Possible Choices: 'group' 'all'
    parser.add_argument('-feat', '--features', dest='feat', type=str, default=None)
    # top num of AU
    parser.add_argument('-feat_num', '--features_num', dest='feat_num', type=int, default=24)

    args = parser.parse_args()

    if args.raw_path is None or not path.isfile(args.raw_path):
        print(r"Input csv doesn't exist: {}".format(args.raw_path))
        exit()

    extract_features(args.raw_path, args.pca,args.au,args.au_num)

"""
import os
os.chdir('C:/Users/owner/Desktop/Project/3deception/')
dsadads
import argparse
import pandas as pd
import os.path as path
from features import utils
raw_df = pd.read_csv('new_csv.csv')
"""