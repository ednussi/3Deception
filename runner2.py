import argparse
from collections import Counter

import numpy as np
import pandas as pd
from os import listdir, path, getcwd
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm
from sklearn.metrics import auc, roc_curve, accuracy_score
from sklearn.model_selection import learning_curve
from features import utils
from learn.utils import cv_method_linear_svm_by_data, prepare_folds
from constants import PCA_METHODS, SPLIT_METHODS, AU_SELECTION_METHODS, FEATURE_SELECTION_METHODS, META_COLUMNS, \
    TARGET_COLUMN, RecordFlags, SUBJECTS

"""
Example Running Command:
python3 runner.py -i 3deception-data/lilach/lilach.csv -a top -an 24 -f all -fn 80 -pm global -p 8 -lm 4v1_A

Arguments:
-i questionnaire/data/alon/fs_shapes.not_quantized.csv -a top -A 51 -f all -F 34 -p global -P 20 -l SP -n max

Mega Runner:
python3 runner.py -i 3deception-data/lilach/lilach.csv -n zero-one -a top -f all -pm global -MR
"""


def parse_preprocessing_params(filename):
    b = filename.split('][')[1:]
    b[-1] = b[-1][:-1]
    return {x[0]: x[1] for x in map(lambda x: x.split('='), b)}


def get_estimator(res_df, test_type, mode='mean_cv'):
    """
    Get best estimator from runs stats.
    
    Mode = 'mean_cv' returns single estimator which gave best test score out of 20 best performing on cross-validation
    Mode = 'all_splits' returns 4 estimators which performed best for every split
    
    Along with each returned estimator, an according pandas series from runs stats is returned.  
    
    :param res_df: results data frame
    :param test_type: test question type to return estimator for
    :param mode: 'mean_cs' or 'all_splits'
    :return: 
    """
    if mode == 'mean_cv':
        # choose best test score out of top 20 best validation scores
        best_res = res_df[res_df.test_type == '[' + str(test_type) + ']'].sort_values(['mean_test_score'],
                                                                                      ascending=False).head(1)
        # best_res = best_res.sort_values(['best_estimator_test_score'], ascending=False).head(1)

        best_estimator = svm.SVC(C=best_res['param_C'].values.tolist()[0], kernel='linear')

        return best_res, best_estimator

    elif mode == 'all_splits':
        results = []
        estimators = []

        for split in range(4):
            # choose best test score out of top 20 best validation scores
            best_res = res_df[res_df.test_type == '[' + str(test_type) + ']'] \
                .sort_values(['split' + str(split) + '_test_score'], ascending=False).head(1)

            results.append(best_res)
            estimators.append(svm.SVC(C=best_res['param_C'].values.tolist()[0], kernel='linear'))

        return results, estimators

    else:
        raise Exception('Unknown mode.')


def prepare_plot(title, x_label, y_label):
    plt.figure()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)


def get_params_dist_single_subject(raw_path, results_path):
    """
    Get accuracy distribution conditioned on AU top-n, features top-n, PCA dimension 
    :param raw_path: 
    :param results_path: 
    :return: 
    """
    if type(results_path) == str:
        res_df = pd.read_csv(results_path, index_col=0)
    else:
        res_df = results_path

    au_top, au_top_acc = [], []
    fe_top, fe_top_acc = [], []
    pca_dims, pca_dims_acc = [], []

    for test_type in range(1, 6):
        res_test = res_df[res_df.test_type == '[' + str(test_type) + ']']

        # AU top-n options and their mean accuracies
        au_top_opts = res_df.au_top.unique().tolist()
        au_mean_accs = []

        for au_top_opt in au_top_opts:
            au_mean_accs.append(res_test[res_test.au_top == au_top_opt].mean_test_score.mean())

        au_top.append(au_top_opts)
        au_top_acc.append(au_mean_accs)

        # Features top-n options and their mean accuracies
        fe_top_opts = res_df.fe_top.unique().tolist()
        fe_mean_accs = []

        for fe_top_opt in fe_top_opts:
            fe_mean_accs.append(res_test[res_test.fe_top == fe_top_opt].mean_test_score.mean())

        fe_top.append(fe_top_opts)
        fe_top_acc.append(fe_mean_accs)

        # PCA dimension options and their mean accuracies
        pca_dim_opts = res_df.pca_dim.unique().tolist()
        pca_dim_mean_accs = []

        for pca_dim_opt in pca_dim_opts:
            pca_dim_mean_accs.append(res_test[res_test.pca_dim == pca_dim_opt].mean_test_score.mean())

        pca_dims.append(pca_dim_opts)
        pca_dims_acc.append(pca_dim_mean_accs)

    return au_top, au_top_acc, fe_top, fe_top_acc, pca_dims, pca_dims_acc


def plot_params_dist_single_subject(raw_path, results_path):
    res_df = pd.read_csv(results_path, index_col=0)

    au_top, au_top_acc, fe_top, fe_top_acc, pca_dims, pca_dims_acc = get_params_dist_single_subject(raw_path, res_df)

    prepare_plot("Top Action Units number (Linear SVM) (" + res_df.learning_method.values[0] + ")", "Top-n AUs",
                 "Score")

    au_top_acc_stack = np.vstack(au_top_acc)
    au_top_acc_mean = au_top_acc_stack.mean(axis=0)
    au_top_acc_std = au_top_acc_stack.std(axis=0)

    plt.fill_between(au_top[0], au_top_acc_mean - au_top_acc_std, au_top_acc_mean + au_top_acc_std,
                     alpha=0.1, color="r")
    plt.plot(au_top[0], au_top_acc_mean, '-', color="g", label="Score")

    plt.savefig(path.join(path.dirname(raw_path), 'au_top_n (' + res_df.learning_method.values[0] + ').png'))

    prepare_plot("Top Features number (Linear SVM) (" + res_df.learning_method.values[0] + ")", "Top-n AUs",
                 "Score")

    fe_top_acc_stack = np.vstack(fe_top_acc)
    fe_top_acc_mean = fe_top_acc_stack.mean(axis=0)
    fe_top_acc_std = fe_top_acc_stack.std(axis=0)

    plt.fill_between(fe_top[0], fe_top_acc_mean - fe_top_acc_std, fe_top_acc_mean + fe_top_acc_std,
                     alpha=0.1, color="r")
    plt.plot(fe_top[0], fe_top_acc_mean, '-', color="g", label="Score")

    plt.savefig(path.join(path.dirname(raw_path), 'features_top_n (' + res_df.learning_method.values[0] + ').png'))

    prepare_plot("PCA dimension (Linear SVM) (" + res_df.learning_method.values[0] + ")", "Top-n AUs",
                 "Score")

    pca_dims_acc_stack = np.vstack(pca_dims_acc)
    pca_dims_acc_mean = pca_dims_acc_stack.mean(axis=0)
    pca_dims_acc_std = pca_dims_acc_stack.std(axis=0)

    plt.fill_between(pca_dims[0], pca_dims_acc_mean - pca_dims_acc_std,
                     pca_dims_acc_mean + pca_dims_acc_std, alpha=0.1, color="r")
    plt.plot(pca_dims[0], pca_dims_acc_mean, '-', color="g", label="Score")

    plt.savefig(path.join(path.dirname(raw_path), 'pca_dims (' + res_df.learning_method.values[0] + ').png'))


def plot_all_params_dist(method='4v1_A'):
    au_tops, fe_tops, pca_dimss = [], [], []
    au_top_accs, fe_top_accs, pca_dims_accs = [], [], []

    for subj in SUBJECTS:
        subj_dir = path.join(getcwd(), 'questionnaire', 'data', subj)
        subj_raw_path = path.join(subj_dir, list(filter(lambda x: x.startswith('fs_'), listdir(subj_dir)))[0])
        subj_results_path = path.join(subj_dir, 'results.{}.csv'.format(method))

        if (path.isfile(subj_results_path)):
            print('-- Processing {}...'.format(subj_results_path))

            au_top, au_top_acc, fe_top, fe_top_acc, pca_dims, pca_dims_acc = \
                get_params_dist_single_subject(subj_raw_path, subj_results_path)

            au_tops.append(au_top)
            fe_tops.append(fe_top)
            pca_dimss.append(pca_dims)
            au_top_accs.append(au_top_acc)
            fe_top_accs.append(fe_top_acc)
            pca_dims_accs.append(pca_dims_acc)

    au_top_accs_stack = np.vstack(au_top_accs)
    fe_top_accs_stack = np.vstack(fe_top_accs)
    pca_dims_accs_stack = np.vstack(pca_dims_accs)

    au_top_accs_stack_mean = au_top_accs_stack.mean(axis=0)
    fe_top_accs_stack_mean = fe_top_accs_stack.mean(axis=0)
    pca_dims_accs_stack_mean = pca_dims_accs_stack.mean(axis=0)

    au_top_accs_stack_std = au_top_accs_stack.std(axis=0)
    fe_top_accs_stack_std = fe_top_accs_stack.std(axis=0)
    pca_dims_accs_stack_std = pca_dims_accs_stack.std(axis=0)

    prepare_plot("Top Action Units number (Linear SVM) (" + method + ")", "Top-n AUs",
                 "Score")

    plt.fill_between(au_top[0], au_top_accs_stack_mean - au_top_accs_stack_std,
                     au_top_accs_stack_mean + au_top_accs_stack_std, alpha=0.1, color="r")
    plt.plot(au_top[0], au_top_accs_stack_mean, '-', color="g", label="Score")

    plt.savefig('all_au_top_n (' + method + ').png')

    prepare_plot("Top Features number (Linear SVM) (" + method + ")", "Top-n AUs",
                 "Score")

    plt.fill_between(fe_top[0], fe_top_accs_stack_mean - fe_top_accs_stack_std,
                     fe_top_accs_stack_mean + fe_top_accs_stack_std, alpha=0.1, color="r")
    plt.plot(fe_top[0], fe_top_accs_stack_mean, '-', color="g", label="Score")

    plt.savefig('all_features_top_n (' + method + ').png')

    prepare_plot("PCA dimension (Linear SVM) (" + method + ")", "Top-n AUs",
                 "Score")

    plt.fill_between(pca_dims[0], pca_dims_accs_stack_mean - pca_dims_accs_stack_std,
                     pca_dims_accs_stack_mean + pca_dims_accs_stack_std, alpha=0.1, color="r")
    plt.plot(pca_dims[0], pca_dims_accs_stack_mean, '-', color="g", label="Score")

    plt.savefig('all_pca_dims (' + method + ').png')


def get_roc_curve_single_subject(raw_path, results_path):
    """
    Calculate average roc curve over all test types for single subject 
    given by FS blendshapes (raw_path) and runs stats (results_path)
    """
    if type(results_path) == str:
        res_df = pd.read_csv(results_path, index_col=0)
    else:
        res_df = results_path

    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)

    fprs, tprs, roc_aucs = [], [], []

    best_fpr, best_tpr, best_roc_auc = None, None, 0.0

    for test_type in range(1, 6):
        # choose best test score out of top 20 best validation scores
        best_res = res_df[res_df.test_type == '[' + str(test_type) + ']'].sort_values(['mean_test_score'],
                                                                                      ascending=False).head(1)
        # best_res = best_res.sort_values(['best_estimator_test_score'], ascending=False).head(1)

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

        train_idx, test_idx = (data_df[data_df.question_type != test_type].index,
                               data_df[data_df.question_type == test_type].index)

        best_estimator = svm.SVC(C=best_res['param_C'].values.tolist()[0], probability=True, kernel='linear')

        class_weight = dict(Counter(target))
        class_weight_sum = max(list(class_weight.values()))

        for x in class_weight.keys():
            class_weight[x] = 1. * class_weight[x] / class_weight_sum

        best_estimator.set_params(class_weight=class_weight)

        best_estimator.fit(data[train_idx, :], target[train_idx])

        y_test = target[test_idx]
        probas = best_estimator.predict_proba(data[test_idx, :])

        # Compute ROC curve and ROC area for each class
        fpr, tpr, thresholds = roc_curve(y_test, probas[:, 1])
        roc_auc = auc(fpr, tpr)

        mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr[0] = 0.0

        fprs.append(fpr)
        tprs.append(tpr)
        roc_aucs.append(roc_auc)

        if roc_auc > best_roc_auc:
            best_fpr = fpr
            best_tpr = tpr
            best_roc_auc = roc_auc

    mean_tpr /= 5
    mean_tpr[-1] = 1.0
    mean_roc_auc = auc(mean_fpr, mean_tpr)

    return fprs, tprs, roc_aucs, mean_fpr, mean_tpr, mean_roc_auc, best_fpr, best_tpr, best_roc_auc


def plot_roc_curve_single_subject(raw_path, results_path):
    res_df = pd.read_csv(results_path, index_col=0)

    prepare_plot("ROC Curves (Linear SVM) (" + res_df.learning_method.values[0] + ")",
                 "False Positive Rate", "True Positive Rate")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    # plot y=x random guess line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    fprs, tprs, roc_aucs, mean_fpr, mean_tpr, mean_roc_auc, _, _, _ = get_roc_curve_single_subject(raw_path, res_df)

    for i, x in enumerate(zip(fprs, tprs, roc_aucs)):
        fpr, tpr, roc_auc = x

        plt.plot(fpr, tpr, color=plt.cm.Paired((2 * i + 1) / 12.),
                 label='ROC curve (test type ' + str(i + 1) + ', area ' + str(roc_auc) + ')')

    plt.plot(mean_fpr, mean_tpr, color='g', linestyle='--', label='ROC curve (mean, area ' + str(mean_roc_auc) + ')')

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig(path.join(path.dirname(raw_path), "roc_curves (" + res_df.learning_method.values[0] + ").png"))


def plot_all_roc_curves(method='4v1_A'):
    mean_fprs, mean_tprs, mean_roc_aucs = [], [], []
    best_fprs, best_tprs, best_roc_aucs = [], [], []

    for subj in SUBJECTS:
        subj_dir = path.join(getcwd(), 'questionnaire', 'data', subj)
        subj_raw_path = path.join(subj_dir, list(filter(lambda x: x.startswith('fs_'), listdir(subj_dir)))[0])
        subj_results_path = path.join(subj_dir, 'results.{}.csv'.format(method))

        if (path.isfile(subj_results_path)):
            print('-- Processing {}...'.format(subj_results_path))

            _, _, _, subj_mean_fpr, subj_mean_tpr, subj_mean_roc_auc, subj_best_fpr, subj_best_tpr, subj_best_roc_auc = \
                get_roc_curve_single_subject(subj_raw_path, subj_results_path)

            mean_fprs.append(subj_mean_fpr)
            mean_tprs.append(subj_mean_tpr)
            mean_roc_aucs.append(subj_mean_roc_auc)

            best_fprs.append(subj_best_fpr)
            best_tprs.append(subj_best_tpr)
            best_roc_aucs.append(subj_best_roc_auc)

    title = "Mean ROC Curves (Linear SVM) (" + method + ")"
    prepare_plot(title, "False Positive Rate", "True Positive Rate")

    # plot y=x random guess line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    for i, x in enumerate(zip(mean_fprs, mean_tprs, mean_roc_aucs)):
        fpr, tpr, roc_auc = x

        plt.plot(fpr, tpr, color=plt.cm.tab20(1. * i / 20),
                 label='ROC curve (subject ' + str(i) + ', area ' + str(roc_auc) + ')')

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('all_mean_roc_curves (' + method + ').png')

    title = "Best ROC Curves (Linear SVM) (" + method + ")"
    prepare_plot(title, "False Positive Rate", "True Positive Rate")

    # plot y=x random guess line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')

    for i, x in enumerate(zip(best_fprs, best_tprs, best_roc_aucs)):
        fpr, tpr, roc_auc = x

        plt.plot(fpr, tpr, color=plt.cm.tab20(1. * i / 20),
                 label='ROC curve (subject ' + str(i) + ', area ' + str(roc_auc) + ')')

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('all_best_roc_curves (' + method + ').png')


def get_learning_curve_single_subject(raw_path, results_path, test_types=range(1, 6)):
    """
    Calculate average learning curve over all test types for single subject 
    given by FS blendshapes (raw_path) and runs stats (results_path)
    """
    if type(results_path) == str:
        res_df = pd.read_csv(results_path, index_col=0)
    else:
        res_df = results_path

    folds = []

    train_scores_means, test_scores_means, train_scores_stds, test_scores_stds = [], [], [], []
    best_train_scores, best_test_scores, best_type, best_test_score_seen = [], [], 0, 0

    for test_type in test_types:
        best_res, best_estimator = get_estimator(res_df, test_type)

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

        folds.append((data_df[data_df.question_type != test_type].index,
                      data_df[data_df.question_type == test_type].index))

        _, train_scores_mean, test_scores_mean, train_scores_std, test_scores_std = \
            get_learning_curve(best_estimator, data, target, ylim=(0.5, 1.01), cv=folds, n_jobs=4)

        train_scores_means.append(train_scores_mean)
        test_scores_means.append(test_scores_mean)
        train_scores_stds.append(train_scores_std)
        test_scores_stds.append(test_scores_std)

    train_sizes = np.linspace(.125, 1.0, 8)

    train_scores_mean_stack = np.vstack(train_scores_means)
    test_scores_mean_stack = np.vstack(test_scores_means)
    train_scores_std_stack = np.vstack(train_scores_stds)
    test_scores_std_stack = np.vstack(test_scores_stds)

    train_scores_mean_fin = train_scores_mean_stack.mean(axis=0)
    test_scores_mean_fin = test_scores_mean_stack.mean(axis=0)
    train_scores_std_fin = train_scores_std_stack.mean(axis=0)
    test_scores_std_fin = test_scores_std_stack.mean(axis=0)

    best_type = np.argmax(test_scores_mean_stack[:, -1])

    best_train_scores = train_scores_means[best_type]
    best_test_scores = test_scores_means[best_type]

    return train_sizes, train_scores_mean_fin, test_scores_mean_fin, train_scores_std_fin, test_scores_std_fin, \
           best_train_scores, best_test_scores, best_type


def plot_learning_curves_single_subject(raw_path, results_path):
    res_df = pd.read_csv(results_path, index_col=0)

    title = "Learning Curves (Linear SVM) (" + res_df.learning_method.values[0] + ")"

    prepare_plot(title, "Training examples", "Score")

    train_sizes, train_scores_mean_fin, test_scores_mean_fin, train_scores_std_fin, test_scores_std_fin, _, _, _ = \
        get_learning_curve_single_subject(raw_path, res_df)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean_fin - train_scores_std_fin,
                     train_scores_mean_fin + train_scores_std_fin, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean_fin - test_scores_std_fin,
                     test_scores_mean_fin + test_scores_std_fin, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean_fin, '-', color="r", label="Train")
    plt.plot(train_sizes, test_scores_mean_fin, '-', color="g", label="Test")

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig(path.join(path.dirname(raw_path), 'learning_curves (' + res_df.learning_method.values[0] + ').png'))


def plot_all_learning_curves(method='4v1_A'):
    train_scores_means, test_scores_means, train_scores_stds, test_scores_stds = [], [], [], []
    best_train_scores, best_test_scores, best_types = [], [], []

    for subj in SUBJECTS:
        subj_dir = path.join(getcwd(), 'questionnaire', 'data', subj)
        subj_raw_path = path.join(subj_dir, list(filter(lambda x: x.startswith('fs_'), listdir(subj_dir)))[0])
        subj_results_path = path.join(subj_dir, 'results.{}.csv'.format(method))

        if (path.isfile(subj_results_path)):
            print('-- Processing {}...'.format(subj_results_path))

            _, subj_train_scores_mean, subj_test_scores_mean, subj_train_scores_std, subj_test_scores_std, \
            subj_best_train_scores, subj_best_test_scores, subj_best_type = \
                get_learning_curve_single_subject(subj_raw_path, subj_results_path)

            train_scores_means.append(subj_train_scores_mean)
            test_scores_means.append(subj_test_scores_mean)
            train_scores_stds.append(subj_train_scores_std)
            test_scores_stds.append(subj_test_scores_std)

            best_train_scores.append(subj_best_train_scores)
            best_test_scores.append(subj_best_test_scores)
            best_types.append(subj_best_type)

    train_sizes = np.linspace(.125, 1.0, 8)

    title = "Mean Learning Curves (Linear SVM) (" + method + ")"
    prepare_plot(title, "Training examples", "Score")
    plt.grid()

    for i, x in enumerate(zip(train_scores_means, test_scores_means, train_scores_stds, test_scores_stds)):
        sm, tsm, ss, tss = x

        plt.fill_between(train_sizes, sm - ss, sm + ss, alpha=0.1, color=plt.cm.tab20(1. * (i + 1) / 20))
        plt.fill_between(train_sizes, tsm - tss, tsm + tss, alpha=0.1, color=plt.cm.tab20(1. * i / 20))

        plt.plot(train_sizes, sm, color=plt.cm.tab20(1. * (i + 1) / 20), label="Train (subject {})".format(i))
        plt.plot(train_sizes, tsm, color=plt.cm.tab20(1. * i / 20), label="Test (subject {})".format(i))

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('all_mean_learning_curves (' + method + ').png')

    title = "Best Learning Curves (Linear SVM) (" + method + ")"
    prepare_plot(title, "Training examples", "Score")
    plt.grid()

    for i, x in enumerate(zip(best_train_scores, best_test_scores, best_types)):
        b, bt, btype = x

        plt.plot(train_sizes, b, color=plt.cm.tab20(1. * (i + 1) / 20),
                 label="Train (subject {}, type {})".format(i, btype))
        plt.plot(train_sizes, bt, color=plt.cm.tab20(1. * i / 20),
                 label="Test (subject {}, type {})".format(i, btype))

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('all_best_learning_curves (' + method + ').png')

    title = "Mean of Mean Learning Curves  (Linear SVM) (" + method + ")"
    prepare_plot(title, "Training examples", "Score")
    plt.grid()

    train_scores_mean_stack = np.vstack(train_scores_means).mean(axis=0)
    test_scores_mean_stack = np.vstack(test_scores_means).mean(axis=0)
    train_scores_std_stack = np.vstack(train_scores_stds).mean(axis=0)
    test_scores_std_stack = np.vstack(test_scores_stds).mean(axis=0)

    plt.fill_between(train_sizes, train_scores_mean_stack - train_scores_std_stack,
                     train_scores_mean_stack + train_scores_std_stack, alpha=0.1, color=plt.cm.tab20(1. * (i + 1) / 20))
    plt.fill_between(train_sizes, test_scores_mean_stack - test_scores_std_stack,
                     test_scores_mean_stack + test_scores_std_stack, alpha=0.1, color=plt.cm.tab20(1. * i / 20))

    plt.plot(train_sizes, train_scores_mean_stack, color="b", label="Train")
    plt.plot(train_sizes, test_scores_mean_stack, color="g", label="Test")

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('mean_of_mean_learning_curves (' + method + ').png')

    title = "Mean of Best Learning Curves  (Linear SVM) (" + method + ")"
    prepare_plot(title, "Training examples", "Score")
    plt.grid()

    best_train_scores_stack = np.vstack(best_train_scores).mean(axis=0)
    best_test_scores_stack = np.vstack(best_test_scores).mean(axis=0)

    plt.plot(train_sizes, best_train_scores_stack, color="b", label="Train")
    plt.plot(train_sizes, best_test_scores_stack, color="g", label="Test")

    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('mean_of_best_learning_curves (' + method + ').png')


def get_learning_curve(estimator, X, y, ylim=None, cv=None, n_jobs=4, train_sizes=np.linspace(.125, 1.0, 8)):
    """
    Generate a simple plot of the test and training learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

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
    if ylim is not None:
        plt.ylim(*ylim)
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    return train_sizes, train_scores_mean, test_scores_mean, train_scores_std, test_scores_std


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
        take_sessions,
        norm='NO'
):
    raw_df = pd.read_csv(raw_path)

    folds, data_fs = prepare_folds(raw_df, learning_method, take_sessions)

    # For each fold
    master_fold_list = []
    for i, fold in enumerate(folds):

        features_path = path.join(path.dirname(raw_path),
                                  "[fold{}][features]{}".format(i, features_params_string))
        au_cor_path = path.join(path.dirname(raw_path),
                                "[fold{}][au-correlation]".format(i, features_params_string))
        features_cor_path = path.join(path.dirname(raw_path),
                                      "[fold{}][features-correlation]".format(i, features_params_string))

        print('In fold {} out of {}'.format(i+1, len(folds)))
        train = fold['train']
        val = fold['val']
        test = fold['test']

        # first train + val indices cover all of the train+val data for this fold
        not_test_indices = [x for x in train[0]['indices']] + [x for x in val[0]['indices']]
        train_val_df = data_fs.iloc[not_test_indices, :]

        # Get AU to use in this fold
        print('Get top AU to be use in fold')
        train_val_au_names = utils.get_top_by_method(train_val_df, au_selection_method, au_top_n,
                                                     feature_selection_method)
        top_au_train_val = utils.return_top_by_features(train_val_df, train_val_au_names.keys())
        if norm != 'NO':
            top_au_train_val = utils.normalize_pd_df(top_au_train_val, norm)

        # Get top Features to use this fold
        print('Get top Features to be use in fold')
        top_features_train_val = utils.get_top_features(top_au_train_val, feature_selection_method, features_top_n,
                                                        learning_method, raw_path)
        train_val_features_names = top_features_train_val.columns[len(META_COLUMNS):]

        # RUN FOR TRAIN & FOLD
        print('Using top AU,Features to run on Train & Val')
        subfold_train_val_list =[]
        for j in range(len(fold['train'])):
            print('In subfold {} out of {}'.format(j + 1, len(fold['train'])))

            fold_sub_train_ind = train[j]['indices']
            fold_sub_val_ind = val[j]['indices']

            fold_train_df = data_fs.iloc[fold_sub_train_ind, :]
            fold_val_df = data_fs.iloc[fold_sub_val_ind, :]

            top_au_train = utils.return_top_by_features(fold_train_df, train_val_au_names.keys())
            top_au_val = utils.return_top_by_features(fold_val_df, train_val_au_names.keys())

            if norm != 'NO':
                top_au_train = utils.normalize_pd_df(top_au_train, norm)
                top_au_val = utils.normalize_pd_df(top_au_val, norm)

            top_features_train = utils.get_top_features_precomputed(top_au_train, train_val_features_names, raw_path)
            top_features_val = utils.get_top_features_precomputed(top_au_val, train_val_features_names, raw_path)

            if learning_method == 'QSP':
                pca_dimension = 10

            if pca_method is not None:
                pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))

                # Combine train and val for PCA
                top_features_train_val = top_features_train.append(top_features_val, ignore_index=True)
                top_features_train_val = utils.dimension_reduction(pca_dimension, pca_method, top_features_train_val)

                # Seperate and save
                top_features_train = top_features_train_val.iloc[:len(top_features_train), :]
                top_features_val = top_features_train_val.iloc[len(top_features_val) + 1:, :]

            subfold_train_val_list.append((top_features_train, top_features_val))

        # RUN FOR TEST
        print('Using top AU,Features to run on Test')

        test_indices = [x for x in test['indices']]
        test_df = data_fs.iloc[test_indices, :]

        print("Choosing Top AU with method:", au_selection_method)
        top_au_test = utils.return_top_by_features(test_df, train_val_au_names.keys())

        if norm != 'NO':
            print("Normalizing with {} method...".format(norm))
            top_au_test = utils.normalize_pd_df(top_au_test, norm)

        print("Extracting features with method:", feature_selection_method)
        top_features_test = utils.get_top_features_precomputed(top_au_test, train_val_features_names, raw_path)

        print("Saving top AU and Features to {} , {} ...".format(au_cor_path, features_cor_path))
        utils.get_corr_(top_au_test, au_top_n, au_selection_method).to_csv(au_cor_path)
        utils.get_corr_(top_features_test, features_top_n, feature_selection_method).to_csv(features_cor_path)

        print("Running PCA...")
        if learning_method == 'QSP':
            pca_dimension = min(top_features_train_val.shape[0],top_features_test.shape[0])

        if pca_method is not None:
            pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))
            top_features_test = utils.dimension_reduction(pca_dimension, pca_method, top_features_test)
            top_features_test.to_csv(pca_path)

        master_fold_list.append((subfold_train_val_list, top_features_test))
        print("Finished fold")
    print("Finished Extract Features")
    return master_fold_list

    #
    #
    #
    #
    #     train_df = data_fs.iloc[train_indices, :]
    #     val_df = data_fs.iloc[val_indices, :]
    #
    #     test_indices = [x for x in test['indices']]
    #     test_df = data_fs.iloc[test_indices, :]
    #
    #     print("Choosing Top AU with method:", au_selection_method)
    #     # top_au_names = utils.get_top_by_method(train_val_df, au_selection_method, au_top_n, learning_method)
    #     #
    #     # # train_df =
    #     # # val_df =
    #     #
    #     # for combo in train:
    #     #     train_indices = [x for x in combo['indices']]
    #     #
    #     #
    #     # Get the top features from correlation of both train and val
    #
    #     top_au_train= utils.return_top_by_features(train_df, train_val_au_names)
    #     top_au_val = utils.return_top_by_features(val_df, train_val_au_names)
    #     top_au_test = utils.return_top_by_features(test_df, train_val_au_names)
    #
    #     # top_au = utils.get_top_n_corr(train_val_df, features_top_n, feature_selection_method)
    #     #
    #     # top_au_train, top_au_test = utils.get_top_au2(train_df, test_df, au_selection_method, au_top_n, learning_method)
    #     # top_au_val, _ = utils.get_top_au2(val_df, test_df, au_selection_method, au_top_n, learning_method)
    #     # top_au_train_val_df , _ = utils.get_top_au2(train_val_df, test_df, au_selection_method, au_top_n, learning_method)
    #
    #
    #     # This is top au for the entire fold
    #     # each fold contain n-1 (4 in 'SP') different train/val combinations
    #
    #     top_au_train_val = top_au_train.append(top_au_val)
    #
    #
    #
    #     if norm != 'NO':
    #         print("Normalizing with {} method...".format(norm))
    #         top_au_train_val = utils.normalize_pd_df(top_au_train_val, norm)
    #         top_au_train = utils.normalize_pd_df(top_au_train, norm)
    #         top_au_val = utils.normalize_pd_df(top_au_val, norm)
    #         top_au_test = utils.normalize_pd_df(top_au_test, norm)
    #
    #     print("Extracting features with method:", feature_selection_method)
    #     top_features_train_val = utils.get_top_features(top_au_train_val, feature_selection_method, features_top_n,learning_method, raw_path)
    #     train_val_features_names = top_features_train_val.columns[len(META_COLUMNS):]
    #
    #     top_features_train = utils.get_top_features_precomputed(top_au_train, train_val_features_names, raw_path)
    #     top_features_val = utils.get_top_features_precomputed(top_au_val, train_val_features_names, raw_path)
    #     top_features_test = utils.get_top_features_precomputed(top_au_test, train_val_features_names, raw_path)
    #
    #     # top_features = utils.get_top_features(top_au, feature_selection_method, features_top_n,
    #     #                                       learning_method, raw_path)
    #     #
    #     # top_features_pd = identifiers.join(df[top_features.keys()])
    #     #
    #     # top_features_test = utils.get_top_features(top_au_test, feature_selection_method, features_top_n,
    #     #                                            learning_method, raw_path)
    #
    #     print("Saving top AU and Features to {} , {} ...".format(au_cor_path, features_cor_path))
    #     utils.get_corr_(top_au_train_val, au_top_n, au_selection_method).to_csv('[train_val]' + au_cor_path)
    #     utils.get_corr_(top_features_train, features_top_n, feature_selection_method).to_csv('[train]' + features_cor_path)
    #
    #     print("Saving all features to {}...".format(features_path))
    #     top_features_train.append(top_features_val).append(top_features_test).to_csv(features_path)
    #
    #     # Cannot produce more n_components then n_samples thus using PCA cannot reduce to a size larger than the min(Samples,Features)
    #     # TODO MORE correct to run on all size of samples and chose minimum
    #     if learning_method == 'QSP':
    #         pca_dimension = min(top_features_train_val.shape[0],top_features_test.shape[0])
    #
    #     if pca_method is not None:
    #         pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))
    #
    #         top_features_train_val = top_features_train.append(top_features_val)
    #         top_features_train_val = utils.dimension_reduction(pca_dimension, pca_method, top_features_train_val)
    #
    #         top_features_train = top_features_train_val.iloc[:len(top_features_train), :]
    #         top_features_val = top_features_train_val.iloc[len(top_features_val) + 1:, :]
    #
    #         top_features_train.to_csv('[train]' + pca_path)
    #         top_features_val.to_csv('[val]' + pca_path)
    #
    #         top_features_test = utils.dimension_reduction(pca_dimension, pca_method, top_features_test)
    #         top_features_test.to_csv('[test]' + pca_path)
    #
    #     master_df = top_features_train.append(top_features_val.append(top_features_test))
    #     top_feat_list.append((top_features_train, top_features_val, top_features_test))
    #
    # return top_feat_list, folds


def cv_method(raw_path, folded_features, method, metric=None, fpstr='', take_sessions=None,
              timestamp=''):
    # folded_features structure:
    #   folded_features = [([(train_df_1_1, val_df_1_1), ...], test_df_1), ...]
    # each [train|val|test]_df contains all metadata columns

    print("Cross validating all learners...")
    results = cv_method_linear_svm_by_data(folded_features, method, metric)

    temp_list = []
    pp_params = parse_preprocessing_params(fpstr)

    results_df = pd.DataFrame(results)

    results_path = path.join(path.dirname(raw_path),
                             'results.{}.{}.{}.csv'.format(timestamp, pp_params['learning-method'], pp_params['norm']))

    with open(results_path, 'a') as f:
        results_df.to_csv(f, header=False)

    return results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # raw input: FS blendshapes time series with some metadata
    parser.add_argument('-i', '--input', dest='raw_path')

    # results csv
    parser.add_argument('-r', '--results', dest='results_path')

    # AU selection method
    parser.add_argument('-a', '--au_selection_method', dest='au_selection_method', type=str, default='top',
                        choices=list(AU_SELECTION_METHODS.values()))

    # Number of top correlated AUs to use if au_choice='top' selected
    parser.add_argument('-A', '--au_top_n', dest='au_top_n', type=int, default=24)

    # Feature selection method
    parser.add_argument('-f', '--feature_selection_method', dest='feature_selection_method', type=str, default=None,
                        choices=list(FEATURE_SELECTION_METHODS.values()))

    # Number of top correlated features to use
    parser.add_argument('-F', '--features_top_n', dest='features_top_n', type=int, default=24)

    # Dimensionality reduction method.
    # 'global' runs PCA on all data
    # 'grouped' runs on each group of features separately (predefined in extract_features())
    parser.add_argument('-p', '--pca_method', dest='pca_method', type=str, default=None,
                        choices=[None] + list(PCA_METHODS.values()))

    # Dimension after reduction.
    # If reduction is global, this is the final dimension
    # If reduction is by groups, this is a dimension for each group
    parser.add_argument('-P', '--pca_dim', dest='pca_dim', type=int, default=None)

    # Folding/learning method
    #
    parser.add_argument('-l', '--learning_method', dest='learning_method', type=str,
                        default=SPLIT_METHODS['4_vs_1_all_session_types'],
                        choices=list(SPLIT_METHODS.values()))

    parser.add_argument('-m', '--metric', dest='metric', type=str, default=None)

    # regular, zero-one, one-one, l1, l2, max
    parser.add_argument('-n', '--norm', dest='norm', type=str, default='NO')

    parser.add_argument('-t', '--take_sessions', dest='take_sessions', type=int, default=None,
                        choices=list(range(2, 21, 2)))

    parser.add_argument('-R', '--mega_runner', dest='mega_runner', action='store_true')

    parser.add_argument('-d', '--draw_plots', dest='draw_plots', action='store_true')

    args = parser.parse_args()

    if args.mega_runner:

        timestamp = time.time()

        def mega_run(raw_path, au_selection_method, feature_selection_method, pca_method,
                     learning_method, metric, take_sessions, norm):

            PCA_RANGE = [25] #range(15, 25)
            FEATURES_RANGE = [25] #range(25, 80, 5)
            AU_RANGE = [51]




            print('Start Mega Run')
            print('Loading CSV...',end='')
            raw_df = pd.read_csv(raw_path)
            print('Done')
            
            print('Preparing Folds...', end='')
            folds, data_fs = prepare_folds(raw_df, learning_method, take_sessions)
            print('Done')

            print('Get top AU to be use in fold')

            for i, fold in enumerate(folds):
                print('In fold {} out of {}'.format(i + 1, len(folds)))

                train = fold['train']
                val = fold['val']
                test = fold['test']

                # first train + val indices cover all of the train+val data for this fold
                not_test_indices = [x for x in train[0]['indices']] + [x for x in val[0]['indices']]
                train_val_df = data_fs.iloc[not_test_indices, :]


                for au_top_n in AU_RANGE:

                    # Get AU to use in this fold
                    print('Get top AU to be use in fold')
                    train_val_au_names = utils.get_top_by_method(train_val_df, au_selection_method, au_top_n,
                                                                 feature_selection_method)
                    top_au_train_val = utils.return_top_by_features(train_val_df, train_val_au_names.keys())

                    if norm != 'NO':
                        top_au_train_val = utils.normalize_pd_df(top_au_train_val, norm)

                    for f_n,features_top_n in enumerate(FEATURES_RANGE):
                        print('In feature {} out of {}'.format(f_n, FEATURES_RANGE))

                        # Get top Features to use this fold
                        print('Get top Features to be use in fold')
                        top_features_train_val = utils.get_top_features(top_au_train_val, feature_selection_method,
                                                                        features_top_n,
                                                                        learning_method, raw_path)
                        train_val_features_names = top_features_train_val.columns[len(META_COLUMNS):]

                        #_______________________RUN FOR TRAIN & FOLD________________________
                        print('Using top AU,Features to run on Train & Val')
                        for p_n,pca_dimension in enumerate(PCA_RANGE):
                            print('In pca dim {} out of {}'.format(p_n, len(PCA_RANGE)))

                            print('Using top AU,Features to run on Train & Val')
                            subfold_train_val_list = []
                            for j in range(len(fold['train'])):
                                print('In subfold {} out of {}'.format(j + 1, len(fold['train'])))

                                fold_sub_train_ind = train[j]['indices']
                                fold_sub_val_ind = val[j]['indices']

                                fold_train_df = data_fs.iloc[fold_sub_train_ind, :]
                                fold_val_df = data_fs.iloc[fold_sub_val_ind, :]

                                top_au_train = utils.return_top_by_features(fold_train_df, train_val_au_names.keys())
                                top_au_val = utils.return_top_by_features(fold_val_df, train_val_au_names.keys())

                                if norm != 'NO':
                                    top_au_train = utils.normalize_pd_df(top_au_train, norm)
                                    top_au_val = utils.normalize_pd_df(top_au_val, norm)

                                top_features_train = utils.get_top_features_precomputed(top_au_train,
                                                                                        train_val_features_names, raw_path)
                                top_features_val = utils.get_top_features_precomputed(top_au_val, train_val_features_names,
                                                                                      raw_path)

                                if learning_method == 'QSP':
                                    pca_dimension = 10

                                if pca_dimension != features_top_n and pca_method is not None:
                                    pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))

                                    # Combine train and val for PCA
                                    top_features_train_val = top_features_train.append(top_features_val,
                                                                                       ignore_index=True)
                                    top_features_train_val = utils.dimension_reduction(pca_dimension, pca_method,
                                                                                       top_features_train_val)

                                    # Seperate and save
                                    top_features_train = top_features_train_val.iloc[:len(top_features_train), :]
                                    top_features_val = top_features_train_val.iloc[len(top_features_val) + 1:, :]

                                subfold_train_val_list.append((top_features_train, top_features_val))


                                #_______________________NAMES________________________
                                features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}]' \
                                                         '[pca-dim={}][pca-method={}][learning-method={}][norm={}]'.format(
                                    path.basename(raw_path),
                                    au_selection_method,
                                    au_top_n,
                                    feature_selection_method,
                                    features_top_n,
                                    pca_dimension,
                                    pca_method,
                                    learning_method,
                                    norm
                                )

                                # Paths Names:
                                features_path = path.join(path.dirname(raw_path),
                                                          "[fold{}][features]{}".format(i, features_params_string))
                                au_cor_path = path.join(path.dirname(raw_path),
                                                        "[fold{}][au-correlation]".format(i, features_params_string))
                                features_cor_path = path.join(path.dirname(raw_path),
                                                              "[fold{}][features-correlation]".format(i,features_params_string))

                            # _______________________RUN FOR TEST________________________
                            print('Using top AU,Features to run on Test')

                            test_indices = [x for x in test['indices']]
                            test_df = data_fs.iloc[test_indices, :]

                            print("Choosing Top AU with method:", au_selection_method)
                            top_au_test = utils.return_top_by_features(test_df, train_val_au_names.keys())

                            if norm != 'NO':
                                print("Normalizing with {} method...".format(norm))
                            top_au_test = utils.normalize_pd_df(top_au_test, norm)

                            print("Extracting features with method:", feature_selection_method)
                            top_features_test = utils.get_top_features_precomputed(top_au_test,
                                                                                   train_val_features_names,
                                                                                   raw_path)

                            print(
                                "Saving top AU and Features to {} , {} ...".format(au_cor_path, features_cor_path))
                            utils.get_corr_(top_au_test, au_top_n, au_selection_method).to_csv(au_cor_path)
                            utils.get_corr_(top_features_test, features_top_n, feature_selection_method).to_csv(
                                features_cor_path)

                            print("Running PCA...")
                            if learning_method == 'QSP':
                                pca_dimension = min(top_features_train_val.shape[0], top_features_test.shape[0])

                            if pca_method is not None:
                                pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))
                            top_features_test = utils.dimension_reduction(pca_dimension, pca_method,
                                                                          top_features_test)
                            top_features_test.to_csv(pca_path)

                            # _______________________EXTRACT FEATURES ENDS HERE________________________
                            cv_method_expected_arg = (subfold_train_val_list, top_features_test)
                            ext_features = cv_method_expected_arg
                            try:
                                cv_method(
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

                            print("Finished One Set of Parameters")
            print("Finished one fold")


            # raw_df = pd.read_csv(args.raw_path)
            #
            # for au_top_n in range(51, 41, -1):
            #     top_au = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)
            #
            #     for features_top_n in range(18, 31):
            #         top_features = utils.get_top_features(top_au, feature_selection_method, features_top_n,
            #                                               learning_method, raw_path)
            #
            #         if pca_method == PCA_METHODS['global']:
            #             pca_options = range(18, 31)
            #         else:
            #             pca_options = range(3, 6)
            #
            #         for pca_dim in pca_options:
            #             try:
            #
            #                 ext_features = utils.dimension_reduction(pca_dim, pca_method, top_features)
            #
            #                 features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}]' \
            #                                          '[pca-dim={}][pca-method={}][learning-method={}][norm={}]'.format(
            #                     path.basename(raw_path),
            #                     au_selection_method,
            #                     au_top_n,
            #                     feature_selection_method,
            #                     features_top_n,
            #                     pca_dim,
            #                     pca_method,
            #                     learning_method,
            #                     norm
            #                 )
            #
            #                 cv_method(
            #                     raw_path,
            #                     ext_features,
            #                     learning_method,
            #                     metric,
            #                     features_params_string,
            #                     take_sessions,
            #                     str(timestamp)
            #                 )
            #             except Exception as e:
            #                 print('----\t[Error]\t' + str(e))


        #### NOT MEGA RUN ####
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

                        mega_run(args.raw_path, au_selection_method, feature_selection_method, pca_method,
                                 learning_method, args.metric, args.take_sessions, args.norm)

                        # take only YES or only NO from all sessions
                        # take only first pair of sessions
                        # try to learn only on second answer (note the dataset may be imbalanced, weight it)

    elif args.draw_plots:
        if args.raw_path is not None and args.results_path is not None:
            plot_learning_curves_single_subject(args.raw_path, args.results_path)
            plot_roc_curve_single_subject(args.raw_path, args.results_path)
            plot_params_dist_single_subject(args.raw_path, args.results_path)

        else:
            plot_all_learning_curves(args.learning_method)
            plot_all_roc_curves(args.learning_method)
            plot_all_params_dist(args.learning_method)

    else:

        if args.pca_method and args.pca_dim is None:
            parser.error("PCA method (-pm/--pca_method) requires dimension (-p/--pca_dim)")
            exit()

        features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}][pca-dim={}]' \
                                 '[pca-method={}][learning-method={}][norm={}]'.format(
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

        folded_features = extract_features(
            args.raw_path,
            args.au_selection_method,
            args.au_top_n,
            args.feature_selection_method,
            args.features_top_n,
            args.pca_method,
            args.pca_dim,
            args.learning_method,
            features_params_string,
            args.take_sessions,
            args.norm
        )

        cv_method(
            args.raw_path,
            folded_features,
            args.learning_method,
            args.metric,
            features_params_string,
            args.take_sessions
        )

#
# """
# import os
# os.chdir('/cs/engproj/3deception/grisha/3deception')
# import argparse
# import pandas as pd
# from os import path
# from features import utils
# raw_path = 'questionnaire/data/omri/fs_shapes.1495611637.0109568.csv'
# features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))
# au_selection_method = 'top'
# au_top_n = 24
# feature_selection_method = 'groups'
# features_top_n = 80
# pca_method = 'groups'
# pca_dim = 6
# learning_method ='SP'
# print("Reading {}...".format(raw_path))
# raw_df = pd.read_csv(raw_path)
# take_sessions = 2
# norm = 'max'
# features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}][pca-dim={}][pca-method={}][learning-method={}][norm={}]'.format(
# path.basename(raw_path),
# au_selection_method,
# au_top_n,
# feature_selection_method,
# features_top_n,
# pca_dim,
# pca_method,
# learning_method,
# norm)

# def prepare_folds(data_df, method='4v1_A', take_sessions=None):
#     """
#     Splits features to list of folds
#     according to learning method.
#
#     :param data_df:
#     :param method:
#     :param take_sessions:
#     :return:
#     """
#
#     # print('-- Data preparation for evaluation method: %s' % method)
#
#     session_types = set_yes_no_types(method)  # take first n sessions
#
#     if take_sessions is not None:
#         data_df = data_df[data_df[SESSION_COLUMN] <= take_sessions]
#
#     # filter answers
#     dfs = []
#
#     temp_df = data_df[data_df[SESSION_TYPE_COLUMN].astype(int) == SESSION_TYPES['say_truth']]
#     dfs.append(temp_df[temp_df[TARGET_COLUMN].astype(int).isin(session_types[SESSION_TYPES['say_truth']])])
#
#     temp_df = data_df[data_df[SESSION_TYPE_COLUMN].astype(int) == SESSION_TYPES['say_lies']]
#     dfs.append(temp_df[temp_df[TARGET_COLUMN].astype(int).isin(session_types[SESSION_TYPES['say_lies']])])
#
#     data_fs = pd.concat(dfs)
#     data_fs.reset_index(inplace=True, drop=True)
#
#     if method == SPLIT_METHODS['4_vs_1_ast_single_answer']:
#         #  duplicate answer with answer_index=1 to answer_index=2
#         data_fs = duplicate_second_answer(data_fs)
#
#     folds = []
#
#     loo = LeaveOneOut()
#
#     question_types = list(QUESTION_TYPES.values())
#
#     if method.startswith('4v1') or method.startswith('SP'):
#         # extract one question type to test on
#         for i, (train_val_types_idx, test_types_idx) in enumerate(loo.split(question_types)):
#
#             test_types = list(map(lambda idx: question_types[idx], test_types_idx))
#
#             # extract frames with test question types
#             test_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(test_types)].index
#
#             train_val_types = list(map(lambda idx: question_types[idx], train_val_types_idx))
#
#             temp_train_folds, temp_val_folds = [], []
#             temp_test_folds = {
#                 'indices': test_indices,
#                 'types': test_types
#             }
#
#             # extract one question type to validate on
#             for j, (train_types_idx, val_types_idx) in enumerate(loo.split(train_val_types)):
#
#                 train_types = list(map(lambda idx: train_val_types[idx], train_types_idx))
#                 val_types = list(map(lambda idx: train_val_types[idx], val_types_idx))
#
#                 # extract frames with train question types
#                 train_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(train_types)].index
#
#                 # extract frames with validation question types
#                 val_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(val_types)].index
#
#                 temp_train_folds.append({
#                     'indices': train_indices,
#                     'types': train_types
#                 })
#
#                 temp_val_folds.append({
#                     'indices': val_indices,
#                     'types': val_types
#                 })
#
#             # add fold dataset
#             folds.append({
#                 'train': temp_train_folds,
#                 'val': temp_val_folds,
#                 'test': temp_test_folds
#             })
#
#     # elif method.startswith('SP'):
#     #     # extract one question type to test on
#     #     for i, (train_val_types_idx, test_types_idx) in enumerate(loo.split(question_types)):
#     #
#     #         test_types = list(map(lambda idx: question_types[idx], test_types_idx))
#     #
#     #         # extract frames with test question types
#     #         test_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(test_types)].index
#     #
#     #         train_val_types = list(map(lambda idx: question_types[idx], train_val_types_idx))
#     #
#     #         temp_train_folds, temp_val_folds = [], []
#     #         temp_test_folds = {
#     #             'indices': test_indices,
#     #             'types': test_types
#     #         }
#     #
#     #         # extract one question type to validate on
#     #         for j, (train_types_idx, val_types_idx) in enumerate(loo.split(train_val_types)):
#     #             train_types = list(map(lambda idx: train_val_types[idx], train_types_idx))
#     #             val_types = list(map(lambda idx: train_val_types[idx], val_types_idx))
#     #
#     #             # extract frames with train question types
#     #             train_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(train_types)].index
#     #
#     #             # extract frames with validation question types
#     #             val_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(val_types)].index
#     #
#     #             temp_train_folds.append({
#     #                 'indices': train_indices,
#     #                 'types': train_types
#     #             })
#     #
#     #             temp_val_folds.append({
#     #                 'indices': val_indices,
#     #                 'types': val_types
#     #             })
#     #
#     #         # add fold dataset
#     #         folds.append({
#     #             'train': temp_train_folds,
#     #             'val': temp_val_folds,
#     #             'test': temp_test_folds
#     #         })
#
#     elif method.startswith('SSP'):
#         sessions_with_types = data_df.loc[:, ['session', 'session_type']].drop_duplicates()
#         session_pairs = product(sessions_with_types[sessions_with_types.session_type == SESSION_TYPES['say_truth']].index,
#                                 sessions_with_types[sessions_with_types.session_type == SESSION_TYPES['say_lies']].index)
#
#         test_sessions_options = [(sessions_with_types.loc[p[0]].session, sessions_with_types.loc[p[1]].session)
#                                  for p in session_pairs]
#
#         # extract one session of each type to test on (2 sessions)
#         for test_sessions in test_sessions_options:
#             # extract frames with test question types
#             test_indices = data_fs[data_fs[SESSION_COLUMN].astype(int).isin(test_sessions)].index
#
#             temp_train_folds, temp_val_folds = [], []
#             temp_test_folds = {
#                 'indices': test_indices,
#                 'types': data_fs.session_type.unique().tolist()  # all session types are in train and in test
#             }
#
#             rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#
#             train_val_df = data_fs[~data_fs[SESSION_COLUMN].astype(int).isin(test_sessions)]
#
#             # extract one question type to validate on
#             for train_idx, val_idx in rs.split(train_val_df):
#
#                 train_types = train_val_df.session_type.unique().tolist()
#                 val_types = train_val_df.session_type.unique().tolist()
#
#                 # extract frames with train question types
#                 train_indices = data_fs.loc[train_idx, :].index
#
#                 # extract frames with validation question types
#                 val_indices = data_fs.loc[val_idx, :].index
#
#                 temp_train_folds.append({
#                     'indices': train_indices,
#                     'types': train_types
#                 })
#
#                 temp_val_folds.append({
#                     'indices': val_indices,
#                     'types': val_types
#                 })
#
#             # add fold dataset
#             folds.append({
#                 'train': temp_train_folds,
#                 'val': temp_val_folds,
#                 'test': temp_test_folds
#             })
#
#     elif method.startswith('QSP'):
#         sessions_with_types = data_df.loc[:, ['session', 'session_type']].drop_duplicates()
#         session_pairs = product(sessions_with_types[sessions_with_types.session_type == SESSION_TYPES['say_truth']].index,
#                                 sessions_with_types[sessions_with_types.session_type == SESSION_TYPES['say_lies']].index)
#
#         test_sessions_options = [(sessions_with_types.loc[p[0]].session, sessions_with_types.loc[p[1]].session)
#                                  for p in session_pairs]
#
#         # extract one session of each type to test on (2 sessions)
#         for test_sessions in test_sessions_options:
#
#             for i, (train_val_types_idx, test_types_idx) in enumerate(loo.split(question_types)):
#                 test_types = list(map(lambda idx: question_types[idx], test_types_idx))
#                 train_val_types = list(map(lambda idx: question_types[idx], train_val_types_idx))
#
#                 # extract frames with test question types
#                 test_session_slice = data_fs[data_fs[SESSION_COLUMN].astype(int).isin(test_sessions)]
#                 test_indices = test_session_slice[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(test_types)].index
#
#                 temp_train_folds, temp_val_folds = [], []
#                 temp_test_folds = {
#                     'indices': test_indices,
#                     'types': test_types
#                 }
#
#                 rs = ShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
#
#                 train_val_df = data_fs[~data_fs[SESSION_COLUMN].astype(int).isin(test_sessions)]
#
#                 # extract one question type to validate on
#                 for train_idx, val_idx in rs.split(train_val_df):
#
#                     # extract one question type to validate on
#                     for j, (train_types_idx, val_types_idx) in enumerate(loo.split(train_val_types)):
#                         train_types = list(map(lambda idx: train_val_types[idx], train_types_idx))
#                         val_types = list(map(lambda idx: train_val_types[idx], val_types_idx))
#
#                         # extract frames with train question types
#                         train_session_slice = data_fs[~data_fs[SESSION_COLUMN].astype(int).isin(test_sessions)]
#                         train_indices = train_session_slice[train_session_slice[QUESTION_TYPE_COLUMN].astype(int).isin(train_types)].index
#
#                         # extract frames with train question types
#                         val_session_slice = data_fs[~data_fs[SESSION_COLUMN].astype(int).isin(test_sessions)]
#                         val_indices = val_session_slice[val_session_slice[QUESTION_TYPE_COLUMN].astype(int).isin(val_types)].index
#
#                         temp_train_folds.append({
#                             'indices': train_indices,
#                             'types': train_types
#                         })
#
#                         temp_val_folds.append({
#                             'indices': val_indices,
#                             'types': val_types
#                         })
#
#                 # add fold dataset
#                 folds.append({
#                     'train': temp_train_folds,
#                     'val': temp_val_folds,
#                     'test': temp_test_folds
#                 })
#
#     else:
#         raise Exception('[prepare_folds] Unknown method: {}'.format(method))
#
#     return folds, data_fs

# def set_yes_no_types(method):
#     session_types = {
#         SESSION_TYPES['say_truth']: [],
#         SESSION_TYPES['say_lies']: []
#     }
#     if method.startswith('4v1'):
#         if method == SPLIT_METHODS['4_vs_1_all_session_types'] or \
#                         method == SPLIT_METHODS['4_vs_1_ast_single_answer']:
#             session_types[SESSION_TYPES['say_truth']] = [
#                 RecordFlags.RECORD_FLAG_ANSWER_TRUE,
#                 RecordFlags.RECORD_FLAG_ANSWER_FALSE
#             ]
#             session_types[SESSION_TYPES['say_lies']] = [
#                 RecordFlags.RECORD_FLAG_ANSWER_TRUE,
#                 RecordFlags.RECORD_FLAG_ANSWER_FALSE
#             ]
#
#         elif method == SPLIT_METHODS['4_vs_1_all_session_types_yes']:
#             # Only learn on answers where "YES" was (should have been) answered
#             session_types[SESSION_TYPES['say_truth']] = [RecordFlags.RECORD_FLAG_ANSWER_TRUE]
#             session_types[SESSION_TYPES['say_lies']] = [RecordFlags.RECORD_FLAG_ANSWER_FALSE]
#
#         elif method == SPLIT_METHODS['4_vs_1_all_session_types_no']:
#             # Only learn on answers where "NO" was (should have been) answered
#             session_types[SESSION_TYPES['say_truth']] = [RecordFlags.RECORD_FLAG_ANSWER_FALSE]
#             session_types[SESSION_TYPES['say_lies']] = [RecordFlags.RECORD_FLAG_ANSWER_TRUE]
#
#         else:
#             raise Exception('Unknown method: %s' % method)
#
#     else:  # session prediction
#         if method == SPLIT_METHODS['session_prediction_leave_qtype_out_yes'] or \
#            method == SPLIT_METHODS['session_prediction_leave_stype_out_yes']:
#             # Only learn on answers where "YES" was (should have been) answered
#             session_types[SESSION_TYPES['say_truth']] = [RecordFlags.RECORD_FLAG_ANSWER_TRUE]
#             session_types[SESSION_TYPES['say_lies']] = [RecordFlags.RECORD_FLAG_ANSWER_FALSE]
#
#         elif method == SPLIT_METHODS['session_prediction_leave_qtype_out_no'] or \
#            method == SPLIT_METHODS['session_prediction_leave_stype_out_no']:
#             # Only learn on answers where "NO" was (should have been) answered
#             session_types[SESSION_TYPES['say_truth']] = [RecordFlags.RECORD_FLAG_ANSWER_FALSE]
#             session_types[SESSION_TYPES['say_lies']] = [RecordFlags.RECORD_FLAG_ANSWER_TRUE]
#
#         else:
#             session_types[SESSION_TYPES['say_truth']] = [
#                 RecordFlags.RECORD_FLAG_ANSWER_TRUE,
#                 RecordFlags.RECORD_FLAG_ANSWER_FALSE
#             ]
#             session_types[SESSION_TYPES['say_lies']] = [
#                 RecordFlags.RECORD_FLAG_ANSWER_TRUE,
#                 RecordFlags.RECORD_FLAG_ANSWER_FALSE
#             ]
#
#     return session_types

# raw_df = pd.read_csv(raw_path)
#
# folds, data_fs = prepare_folds(raw_df, learning_method, take_sessions)
#
# # For each fold
# top_feat_list = []
#
# for i, fold in enumerate(folds):
#     features_path = path.join(path.dirname(raw_path),
#                               "[fold{}][features]{}".format(i, features_params_string))
#     au_cor_path = path.join(path.dirname(raw_path),
#                             "[fold{}][au-correlation]".format(i, features_params_string))
#     features_cor_path = path.join(path.dirname(raw_path),
#                                   "[fold{}][features-correlation]".format(i, features_params_string))
#
#     print('In fold {} out of {}'.format(i, len(folds)))
#     train = fold['train']
#     val = fold['val']
#     test = fold['test']
#
#     not_test_indices = [x for x in train[0]['indices']] + [x for x in val[0]['indices']]
#     train_val_df = data_fs.iloc[not_test_indices, :]
#
#     print("Choosing Top AU with method", au_selection_method)
#     top_au, top_au_test = utils.get_top_au2(train_val_df, test, au_selection_method, au_top_n, learning_method)
