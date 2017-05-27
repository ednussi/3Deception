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


def cv_method_all_learners(raw_path, ext_features, method, metric=None, features_params_string='', take_sessions=None, timestamp=''):
    # print("Cross validating all learners...")
    results = cv_method_all_classifiers(ext_features, method, metric, take_sessions)

    temp_list = []

    pp_params = parse_preprocessing_params(features_params_string)
    
    for x in results:
        temp_df = pd.DataFrame(x['cv_results'].cv_results_)
        temp_df['best_mean_val_score'] = x['best_mean_val_score']
        temp_df['train_types'] = str(x['train_types']).replace(',', '')
        temp_df['val_type'] = str(x['val_type']).replace(',', '')
        temp_df['test_type'] = str(x['test_type']).replace(',', '')
        temp_df['best_estimator_train_score'] = x['best_estimator_train_score']
        temp_df['best_estimator_test_score'] = x['best_estimator_test_score']
        temp_df['au-method'] = pp_params['au-method']
        temp_df['au-top-n'] = pp_params['au-top-n']
        temp_df['fe-method'] = pp_params['fe-method']
        temp_df['fe-top-n'] = pp_params['fe-top-n']
        temp_df['learning-method'] = pp_params['learning-method']
        temp_df['pca-dim'] = pp_params['pca-dim']
        temp_df['pca-method'] = pp_params['pca-method']

        temp_list.append(temp_df.join(pd.DataFrame([method] * len(temp_df), columns=['method'])))

    results_df = pd.concat(temp_list)

    results_path = path.join(path.dirname(raw_path), "results." + timestamp + '.csv')

    with open(results_path, 'a') as f:
        results_df.to_csv(f, header=False)

    return results_df


def parse_preprocessing_params(filename):
    b = filename.split('][')[1:]
    b[-1] = b[-1][:-1]
    return {x[0]: x[1] for x in map(lambda x: x.split('='), b)}


def get_roc_curves(raw_path, results_path):
    raw_df = pd.read_csv(raw_path)
    res_df = pd.read_csv(results_path, index_col=0)

    title = "ROC Curves (Linear SVM)"

    plt.figure()
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(title)

    for test_type in range(1,6):
        # choose best test score out of top 20 best validation scores
        best_res = res_df[res_df.test_type == '[' + str(test_type) + ']'].sort_values(['mean_test_score'], ascending=False).head(20)
        best_res = best_res.sort_values(['best_estimator_test_score'], ascending=False).head(1)

        best_estimator = svm.SVC(C=best_res['param_C'].values.tolist()[0], kernel='linear')

        data_df = extract_features(
            raw_path,
            best_res['au_method'].values.tolist()[0],
            int(best_res['au_top'].values.tolist()[0]),
            best_res['fe_method'].values.tolist()[0],
            int(best_res['fe_top'].values.tolist()[0]),
            best_res['pca_method'].values.tolist()[0],
            int(best_res['pca_dim'].values.tolist()[0]),
            best_res['learning_method'].values.tolist()[0],
            '.garbage'
        )

        data = data_df.iloc[:, len(META_COLUMNS):].values
        target = (data_df[TARGET_COLUMN] == RecordFlags.RECORD_FLAG_ANSWER_TRUE).values

        train_idx, test_idx = (data_df[data_df.question_type != test_type].index, data_df[data_df.question_type == test_type].index)

        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in [True, False]: TODO
            fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # lw = 2
        # plt.plot(fpr[2], tpr[2], color='darkorange',
        #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
        # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        # plt.xlim([0.0, 1.0])
        # plt.ylim([0.0, 1.05])
        # plt.legend(loc="lower right")
        # plt.show()

    plt.legend(loc='best', prop={'size':6})
    plt.savefig(path.join(path.dirname(raw_path), 'learning_curves.png'))


def get_learning_curves(raw_path, results_path):
    raw_df = pd.read_csv(raw_path)
    res_df = pd.read_csv(results_path, index_col=0)

    title = "Learning Curves (Linear SVM)"

    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    for test_type in range(1,6):
        # choose best test score out of top 20 best validation scores
        best_res = res_df[res_df.test_type == '[' + str(test_type) + ']'].sort_values(['mean_test_score'], ascending=False).head(20)
        best_res = best_res.sort_values(['best_estimator_test_score'], ascending=False).head(1)

        best_estimator = svm.SVC(C=best_res['param_C'].values.tolist()[0], kernel='linear')

        data_df = extract_features(
            raw_path,
            best_res['au_method'].values.tolist()[0],
            int(best_res['au_top'].values.tolist()[0]),
            best_res['fe_method'].values.tolist()[0],
            int(best_res['fe_top'].values.tolist()[0]),
            best_res['pca_method'].values.tolist()[0],
            int(best_res['pca_dim'].values.tolist()[0]),
            best_res['learning_method'].values.tolist()[0],
            '.garbage'
        )

        data = data_df.iloc[:, len(META_COLUMNS):].values
        target = (data_df[TARGET_COLUMN] == RecordFlags.RECORD_FLAG_ANSWER_TRUE).values

        folds = [(data_df[data_df.question_type != test_type].index, data_df[data_df.question_type == test_type].index)]

        plot_learning_curve(best_estimator, title, data, target, ylim=(0.5, 1.01), cv=folds,
                            n_jobs=4, raw_path=raw_path, 
                            output_path='best_learning_curve.test_type_' + str(test_type) + '.png',
                            test_type=test_type)

    plt.legend(loc='best', prop={'size':6})
    plt.savefig(path.join(path.dirname(raw_path), 'learning_curves.png'))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.125, 1.0, 8), raw_path='',
                        output_path='best_learning_curve.png', test_type=0):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    # plt.figure()
    # plt.title(title)
    # plt.xlabel("Training examples")
    # plt.ylabel("Score")
    if ylim is not None:
        plt.ylim(*ylim)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color=plt.cm.Paired(test_type / 6.),
             label="Train, type " + str(test_type))
    plt.plot(train_sizes, test_scores_mean, 'o-', color=plt.cm.Paired((1. + test_type) / 6. - 1./12),
             label="Test, type " + str(test_type))

    # plt.legend(loc="best")

    # plt.savefig(path.join(path.dirname(raw_path), output_path))
    return plt


def extract_features(
        raw_path,
        au_selection_method,
        au_top_n,
        feature_selection_method,
        features_top_n,
        pca_method,
        pca_dimension,
        learning_method,
        features_params_string,
        norm
):

    features_path = path.join(path.dirname(raw_path), "features__" + features_params_string)
    au_cor_path = path.join(path.dirname(raw_path), "au-correlation__" + features_params_string)
    features_cor_path = path.join(path.dirname(raw_path), "features-correlation__" + features_params_string)

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    if norm != 'NO':
        print("Normalizing with {} method...".format(norm))
        raw_df = utils.normalize_pd_df(raw_df, norm)

    print("Choosing Top AU with method", au_selection_method)
    top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)

    print("Extracting features with method:", feature_selection_method)
    top_features = utils.get_top_features(top_AU, feature_selection_method, features_top_n, learning_method, raw_path)

    print("Saving top AU and Features to {} , {} ...".format(au_cor_path, features_cor_path))
    utils.get_corr_(top_AU, au_top_n, au_selection_method).to_csv(au_cor_path)
    utils.get_corr_(top_features, features_top_n, feature_selection_method).to_csv(features_cor_path)

    print("Saving all features to {}...".format(features_path))
    top_features.to_csv(features_path)

    if pca_method is not None:
        pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))

        top_features = utils.dimension_reduction(pca_dimension, pca_method, top_features)
        top_features.to_csv(pca_path)

    return top_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # raw input: FS blendshapes time series with some metadata
    parser.add_argument('-i', '--input', dest='raw_path', required=True)

    # results csv
    parser.add_argument('-r', '--results', dest='results_path')

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

    parser.add_argument('-n', '--norm', dest='norm', type=str, default='NO')

    parser.add_argument('-ts', '--take_sessions', dest='take_sessions', type=int, default=None,
                        choices=list(range(2, 21, 2)))

    parser.add_argument('-MR', '--mega_runner', dest='mega_runner', action='store_true')

    parser.add_argument('-SH', '--show_results', dest='show_results', action='store_true')

    args = parser.parse_args()

    if args.mega_runner:

        timestamp = time.time()

        def mega_run(raw_path, au_selection_method, feature_selection_method, pca_method,
                     learning_method, metric, take_sessions, norm):
            raw_df = pd.read_csv(args.raw_path)

            if norm != 'NO':
                print("Normalizing with {} method...".format(norm))
                raw_df = utils.normalize_pd_df(raw_df, norm)

            for au_top_n in range(24, 15, -1):
                top_au = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)

                for features_top_n in range(20, 90, 2):
                    top_features = utils.get_top_features(top_au, feature_selection_method, features_top_n,
                                                          learning_method, raw_path)

                    if pca_method == PCA_METHODS['global']:
                        pca_options = range(30, 6, -1)
                    else:
                        pca_options = range(3, 6)

                    for pca_dim in pca_options:
                        try:

                            ext_features = utils.dimension_reduction(pca_dim, pca_method, top_features)

                            features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}][pca-dim={}][pca-method={}][learning-method={}][normalization={}]'.format(
                                path.basename(raw_path),
                                au_selection_method,
                                au_top_n,
                                feature_selection_method,
                                features_top_n,
                                pca_dim,
                                pca_method,
                                learning_method,
                                norm
                            )

                            cv_method_all_learners(
                                raw_path,
                                ext_features,
                                learning_method,
                                metric,
                                features_params_string,
                                take_sessions,
                                str(timestamp)
                            )
                        except Exception as e:
                            print('----\t[Error]\t' + str(e))

        if args.au_selection_method is None:
            print('No AU selection method provided')
            pass

        else:  # use given au selection method
            au_selection_method = args.au_selection_method

            if args.feature_selection_method is None:
                print('No feature selection method provided')
                pass

            else:  # use given feature selection method
                feature_selection_method = args.feature_selection_method

                if args.pca_method is None:
                    print('No PCA method provided')
                    pass

                else:  # use given pca method
                    pca_method = args.pca_method

                    if args.learning_method is None:
                        print('No learning method provided')
                        pass

                    else:  # use given learning method
                        learning_method = args.learning_method

                        mega_run(args.raw_path, au_selection_method, feature_selection_method, pca_method, learning_method, args.metric, args.take_sessions, args.norm)

        # take only YES or only NO from all sessions
        # take only first pair of sessions
        # try to learn only on second answer (note the dataset may be imbalanced, weight it)

    elif args.show_results:
        if args.results_path is None:
            print('Results path not provided')
            exit(1)

        get_learning_curves(args.raw_path, args.results_path)

    else:

        if args.pca_method and args.pca_dim is None:
            parser.error("PCA method (-pm/--pca_method) requires dimension (-p/--pca_dim)")
            exit()

        features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}][pca-dim={}][pca-method={}][learning-method={}][normalization={}]'.format(
            path.basename(args.raw_path),
            args.au_selection_method,
            args.au_top_n,
            args.feature_selection_method,
            args.features_top_n,
            args.pca_dim,
            args.pca_method,
            args.learning_method,
            args.norm
        )

        ext_features = extract_features(
            args.raw_path,
            args.au_selection_method,
            args.au_top_n,
            args.feature_selection_method,
            args.features_top_n,
            args.pca_method,
            args.pca_dim,
            args.learning_method,
            features_params_string,
            args.norm
        )

        cv_method_all_learners(
            args.raw_path,
            ext_features,
            args.learning_method,
            args.metric,
            features_params_string,
            args.take_sessions
        )

#"""
#import os
#os.chdir('/cs/engproj/3deception/eran/3deception')
#import argparse
#import pandas as pd
#import os.path as path
#from features import utils
#raw_path = 'new_jonathan.csv'
#features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))
#au_selection_method = 'top'
#au_top_n = 24
#feature_selection_method = 'groups'
#features_top_n = 80
#pca_method = 'groups'
#pca_dim = 6
#learning_method ='4v1_T'
#print("Reading {}...".format(raw_path))
#raw_df = pd.read_csv(raw_path)
#print("Choosing Top AU with method", au_selection_method)
#top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)
#print("Extracting features with method:", feature_selection_method)
#top_features = utils.get_top_features(top_AU, feature_selection_method, features_top_n, learning_method, raw_path)
#print("Saving all features to {}...".format(features_path), end="")
#top_features.to_csv(features_path)
#"""
