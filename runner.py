import argparse
import numpy as np
import pandas as pd
import os.path as path
from features import utils
from learn.utils import cv_method_all_classifiers

import matplotlib
matplotlib.use('Agg')

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import operator as o
from constants import PCA_METHODS, SPLIT_METHODS, AU_SELECTION_METHODS, FEATURE_SELECTION_METHODS


def cv_method_all_learners(raw_path, features_path, method, metric=None):
    print("Cross validating all learners...")
    results = cv_method_all_classifiers(features_path, method, metric)

    temp_list = []
    for x in results:
        temp_df = pd.DataFrame(x['cv_results'].cv_results_)
        temp_list.append(temp_df.join(pd.DataFrame([method] * len(temp_df), columns=['method'])))

    results_df = pd.concat(temp_list)

    results_path = path.join(path.dirname(raw_path), "learning-results_" + path.basename(raw_path))
    print("Saving learning results to {}...".format(results_path))
    results_df.to_csv(results_path)

    plot_path = results_path.replace('.csv', '.{}.png'.format(method))
    print("Plotting learning results to {}...".format(plot_path))
    save_results_plot(plot_path, results)


def save_results_plot(plot_path, results):
    def barplot(ax, dpoints):
        """
        Create a barchart for data across different categories with
        multiple conditions for each category.
        
        @param ax: The plotting axes from matplotlib.
        @param dpoints: The data set as an (n, 3) numpy array
        """

        # Aggregate the conditions and the categories according to their
        # mean values
        conditions = [(c, np.mean(dpoints[dpoints[:, 0] == c][:, 2].astype(float)))
                      for c in np.unique(dpoints[:, 0])]
        categories = [(c, np.mean(dpoints[dpoints[:, 1] == c][:, 2].astype(float)))
                      for c in np.unique(dpoints[:, 1])]

        # sort the conditions, categories and data so that the bars in
        # the plot will be ordered by category and condition
        conditions = [c[0] for c in sorted(conditions, key=o.itemgetter(1))]
        categories = [c[0] for c in sorted(categories, key=o.itemgetter(1))]

        dpoints = np.array(sorted(dpoints, key=lambda x: categories.index(x[1])))

        # the space between each set of bars
        space = 0.3
        n = len(conditions)
        width = (1 - space) / (len(conditions))

        # Create a set of bars at each position
        for i, cond in enumerate(conditions):
            indeces = range(1, len(categories) + 1)
            vals = dpoints[dpoints[:, 0] == cond][:, 2].astype(np.float)
            pos = [j - (1 - space) / 2. + i * width for j in indeces]
            ax.bar(pos, vals, width=width, label=cond,
                   color=cm.Accent(float(i) / n))

        # Set the x-axis tick labels to be equal to the categories
        ax.set_xticks(indeces)
        ax.set_xticklabels(categories)
        plt.setp(plt.xticks()[1])

        # Add the axis labels
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Classifiers")

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left')

        plt.savefig(plot_path)

    data = []
    for res in results:
        estimator = res['estimator']
        res_df = pd.DataFrame(res['cv_results'].cv_results_).sort(['mean_test_score'], ascending=[0])

        train_score = res_df.loc[0, :]['mean_train_score']
        test_score = res_df.loc[0, :]['mean_test_score']

        data.append([estimator, 'best_mean_train_score', train_score])
        data.append([estimator, 'best_mean_test_score', test_score])

    data = np.array(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    barplot(ax, data)


def extract_features(
        raw_path,
        au_selection_method,
        au_top_n,
        feature_selection_method,
        features_top_n,
        pca_method,
        pca_dimension,
        learning_method
):
    features_params = 'i_{}_a_{}_an_{}_f_{}_fn_{}_p_{}_pm_{}_m_{}'.format(raw_path,au_selection_method,au_top_n,feature_selection_method,
                                                                          features_top_n,pca_dimension,pca_method,learning_method)


    features_path = path.join(path.dirname(raw_path), features_params + "_features_" + path.basename(raw_path))
    au_cor_path = path.join(path.dirname(raw_path), features_params + "_au_correlation_" + path.basename(raw_path))
    features_cor_path = path.join(path.dirname(raw_path), features_params + "_features_correlation_" + path.basename(raw_path))

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Choosing Top AU with method", au_selection_method)
    top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)

    print("Extracting features with method:", feature_selection_method)
    top_features = utils.get_top_features(top_AU, feature_selection_method, features_top_n, learning_method, raw_path)

    print("Saving top AU and Features to {} , {} ...".format(au_cor_path, features_cor_path), end="")
    utils.get_cor(top_AU, au_top_n, au_selection_method).to_csv(au_cor_path)
    utils.get_cor(top_features, features_top_n, feature_selection_method).to_csv(features_cor_path)

    print("Saving all features to {}...".format(features_path), end="")
    top_features.to_csv(features_path)

    return_path = features_path

    if pca_method is not None:
        pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))

        utils.dimension_reduction(pca_dimension, pca_method, pca_path, top_features)

        return_path = pca_path

    return return_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='raw_path', required=True)

    # AU selection method
    parser.add_argument('-a', '--au_selection_method', dest='au_selection_method', type=str, default='top',
                        choices=list(AU_SELECTION_METHODS.values()))

    # Number of top correlated AUs to use if au_choice='top' selected
    parser.add_argument('-an', '--au_top_n', dest='au_top_n', type=int, default=24)

    # Feature selection method
    parser.add_argument('-f', '--feature_selection_method', dest='feature_selection_method', type=str, default=None,
                        choices=list(FEATURE_SELECTION_METHODS.values()))

    # Number of top correlated features to use
    parser.add_argument('-fn', '--features_top_n', dest='features_top_n', type=int, default=24)

    # Dimensionality reduction method.
    # 'global' runs PCA on all data
    # 'grouped' runs on each group of features separately (predefined in extract_features())
    parser.add_argument('-pm', '--pca_method', dest='pca_method', type=str, default=None,
                        choices=[None] + list(PCA_METHODS.values()))

    # Dimension after reduction.
    # If reduction is global, this is the final dimension
    # If reduction is by groups, this is a dimension for each group
    parser.add_argument('-p', '--pca_dim', dest='pca_dim', type=int, default=None)

    # Folding/learning method
    parser.add_argument('-lm', '--learning_method', dest='learning_method', type=str,
                        default=SPLIT_METHODS['4_vs_1_all_session_types'],
                        choices=list(SPLIT_METHODS.values()))

    parser.add_argument('-m', '--metric', dest='metric', type=str, default=None)

    args = parser.parse_args()

    if args.pca_method and args.pca_dim is None:
        parser.error("PCA method (-pm/--pca_method) requires dimension (-p/--pca_dim)")
        exit()

    # features_path = extract_features(
    #     args.raw_path,
    #     args.au_selection_method,
    #     args.au_top_n,
    #     args.feature_selection_method,
    #     args.features_top_n,
    #     args.pca_method,
    #     args.pca_dim,
    #     args.learning_method
    # )
    features_path = 'pca_new_jonathan.csv'

    cv_method_all_learners(
        args.raw_path,
        features_path,
        args.learning_method,
        args.metric
    )

"""
import os
os.chdir('/cs/engproj/3deception/eran/3deception')
import argparse
import pandas as pd
import os.path as path
from features import utils
raw_path = 'new_jonathan.csv'
features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))
au_selection_method = 'top'
au_top_n = 24
feature_selection_method = 'groups'
features_top_n = 80
pca_method = 'groups'
pca_dim = 6
learning_method ='4v1_T'
print("Reading {}...".format(raw_path))
raw_df = pd.read_csv(raw_path)
print("Choosing Top AU with method", au_selection_method)
top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)
print("Extracting features with method:", feature_selection_method)
top_features = utils.get_top_features(top_AU, feature_selection_method, features_top_n, learning_method, raw_path)
print("Saving all features to {}...".format(features_path), end="")
top_features.to_csv(features_path)
"""
