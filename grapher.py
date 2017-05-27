import argparse
import numpy as np
import pandas as pd
from os import listdir, path
import time
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.model_selection import learning_curve
from features import utils
from learn.utils import cv_method_all_classifiers, prepare_folds
from constants import PCA_METHODS, SPLIT_METHODS, AU_SELECTION_METHODS, FEATURE_SELECTION_METHODS, META_COLUMNS, TARGET_COLUMN, RecordFlags


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    print('Reading arguments..')
    # results csv
    parser.add_argument('-r', '--results', dest='results_path')

    args = parser.parse_args()

    print('Loading csv..')
    res_df = pd.read_csv(args.results_path, index_col=0)

    print('Plotting graphs..')

    ######################### MULTI DIM GRAPH #########################
    fig = plt.figure(1)
    ax = fig.add_subplot(111, projection='3d')

    # Defined by the accuracy
    size = res_df['best_estimator_test_score']*2 - 1

    # Defined by the C parameter
    color = plt.cm.Reds(utils.normalize_pd_df(res_df['param_C'], 'zero-one'))

    ax.scatter(res_df['au_top'], res_df['fe_top'], res_df['pca_dim'], s=size, c=color)

    ax.set_xlabel('Top Action Units Number')
    ax.set_ylabel('Top Features Number')
    ax.set_zlabel('PCA Dimension Size')

    plt.show()

    fig_name = args.raw_path + '_MULTI_GRAPH.png'
    print('Saving fig into:{}..'.format(fig_name))
    plt.savefig(fig_name)

    print('Grapher Finished!')

    ######################### MULTI DIM GRAPH #########################



