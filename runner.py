import argparse
import numpy as np
import pandas as pd
import os.path as path
from features import utils
from learn.utils import cv_method_all_classifiers
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import operator as o


# TODO
#     AU SELECTION
#     FEATURE SELECTION (Groups vs global, dimension (size))
#     PCA (Groups vs global, dimension (size))


def cv_method_all_learners(raw_path, features_path, method, metric=None):
    print("Cross validating all learners...")
    results = cv_method_all_classifiers(features_path, method, metric)

    results_df = pd.concat([pd.DataFrame(x['results'], index=x['method']) for x in results])

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
        plt.setp(plt.xticks()[1], rotation=90)

        # Add the axis labels
        ax.set_ylabel("RMSD")
        ax.set_xlabel("Structure")

        # Add a legend
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], loc='upper left')

        plt.savefig(plot_path)

    data = []
    for res in mres['results']:
        estimator = res['estimator']
        train_score = res['cv_results']['mean_train_score'].values.tolist()
        test_score = res['cv_results']['mean_test_score'].values.tolist()

        data.append([estimator, 'mean_train_score', train_score[0]])
        data.append([estimator, 'mean_test_score', test_score[0]])

    data = np.array(data)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    barplot(ax, data)


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

    return fpath


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
    
    parser.add_argument('-m', '--method', dest='method', type=str, default='4v1_A', chocies=['4v1_A','4v1_T','4v1_F','SP'])

    parser.add_argument('-mr', '--metric', dest='metric', type=str, default=None)


    args = parser.parse_args()

    if args.raw_path is None or not path.isfile(args.raw_path):
        print(r"Input csv doesn't exist: {}".format(args.raw_path))
        exit()

    features_path = extract_features(
        args.raw_path, 
        args.pca,
        args.au,
        args.au_num,
        args.feat,
        args.feat_num,
        args.method
        )

    cv_method_all_learners(
        args.raw_path, 
        features_path, 
        args.method,
        metric=None
        )



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