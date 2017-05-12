import argparse
import pandas as pd
import os.path as path
from features import utils
from learn.utils import cv_all_methods



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


def extract_features(raw_path, with_pca, au, au_num, feat, feat_num, method):
    features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Choosing Top AU with method",au)
    top_AU = utils.get_top_au(raw_df, au, au_num, method)

    print("Extracting features with method:",feat)
    top_features = utils.get_top_features(top_AU, feat,feat_num, method)

    print("Saving all features to {}...".format(features_path), end="")
    top_features.to_csv(features_path)

    fpath = features_path

    if with_pca is not None:
        pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))
        print("Running PCA...")
        pca_features = utils.pca_3d(top_features, with_pca)
        print("Saving PCA features to {}...".format(pca_path), end="")
        pca_features.to_csv(pca_path)

        fpath = pca_path

    print("Learning...")
    results_df = cv_all_methods(fpath)

    results_path = path.join(path.dirname(raw_path), "learning-results_" + path.basename(raw_path))
    print("Saving learning results to {}...".format(results_path))
    results_df.to_csv(results_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='raw_path')
    parser.add_argument('-p', '--pca', dest='pca', type=int, default=None)

    # Possible Choices: 'daniel' 'mouth' 'eyes' 'brows' 'eyes_area' 'smile' 'blinks' 'top'
    parser.add_argument('-a', '--au_choice', dest='au', type=str, default=None)
    # If au was chosen 'top' need a number for number of top AU
    parser.add_argument('-an', '--au_top_num', dest='au_num', type=int, default=24)

    # Possible Choices: 'group' 'all'
    parser.add_argument('-f', '--features', dest='feat', type=str, default=None)
    # top num of AU
    parser.add_argument('-fn', '--features_num', dest='feat_num', type=int, default=24)
    # Choices:
    parser.add_argument('-m', '--method', dest='method', type=str, default='4v1_A', chocies=['4v1_A','4v1_T','4v1_F','SP'])


    args = parser.parse_args()

    if args.raw_path is None or not path.isfile(args.raw_path):
        print(r"Input csv doesn't exist: {}".format(args.raw_path))
        exit()

    extract_features(args.raw_path, args.pca,args.au,args.au_num,args.feat,args.feat_num,args.method)

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