import argparse
from collections import Counter

import numpy as np
import pandas as pd
from os import listdir, path, getcwd
import time
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import svm
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.model_selection import learning_curve
from sklearn.metrics import accuracy_score
from features import utils
from learn.utils import cv_method_all_classifiers, prepare_folds
from constants import PCA_METHODS, SPLIT_METHODS, AU_SELECTION_METHODS, FEATURE_SELECTION_METHODS, META_COLUMNS, \
    TARGET_COLUMN, RecordFlags, SUBJECTS

"""
Example Running Command:
python3 runner.py -i 3deception-data/lilach/lilach.csv -a top -an 24 -f all -fn 80 -pm global -p 8 -lm 4v1_A

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
        best_res = res_df[res_df.test_type == '[' + str(test_type) + ']'].sort_values(['mean_test_score'], ascending=False).head(1)
        # best_res = best_res.sort_values(['best_estimator_test_score'], ascending=False).head(1)

        best_estimator = svm.SVC(C=best_res['param_C'].values.tolist()[0], kernel='linear')

        return best_res, best_estimator

    elif mode == 'all_splits':
        results = []
        estimators = []

        for split in range(4):
            # choose best test score out of top 20 best validation scores
            best_res = res_df[res_df.test_type == '[' + str(test_type) + ']']\
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
    
    plt.fill_between(au_top[0], au_top_acc_mean - au_top_acc_std, au_top_acc_mean + au_top_acc_std, alpha=0.1, color="r")
    plt.plot(au_top[0], au_top_acc_mean, '-', color="g", label="Score")

    plt.savefig(path.join(path.dirname(raw_path), 'au_top_n (' + res_df.learning_method.values[0] + ').png'))

    prepare_plot("Top Features number (Linear SVM) (" + res_df.learning_method.values[0] + ")", "Top-n AUs",
                 "Score")

    fe_top_acc_stack = np.vstack(fe_top_acc)
    fe_top_acc_mean = fe_top_acc_stack.mean(axis=0)
    fe_top_acc_std = fe_top_acc_stack.std(axis=0)
    
    plt.fill_between(fe_top[0], fe_top_acc_mean - fe_top_acc_std, fe_top_acc_mean + fe_top_acc_std, alpha=0.1, color="r")
    plt.plot(fe_top[0], fe_top_acc_mean, '-', color="g", label="Score")

    plt.savefig(path.join(path.dirname(raw_path), 'features_top_n (' + res_df.learning_method.values[0] + ').png'))

    prepare_plot("PCA dimension (Linear SVM) (" + res_df.learning_method.values[0] + ")", "Top-n AUs",
                 "Score")

    pca_dims_acc_stack = np.vstack(pca_dims_acc)
    pca_dims_acc_mean = pca_dims_acc_stack.mean(axis=0)
    pca_dims_acc_std = pca_dims_acc_stack.std(axis=0)
    
    plt.fill_between(pca_dims[0], pca_dims_acc_mean - pca_dims_acc_std, pca_dims_acc_mean + pca_dims_acc_std, alpha=0.1, color="r")
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

            au_top, au_top_acc, fe_top, fe_top_acc, pca_dims, pca_dims_acc = get_params_dist_single_subject(subj_raw_path, subj_results_path)

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

    plt.fill_between(au_top[0], au_top_accs_stack_mean - au_top_accs_stack_std, au_top_accs_stack_mean + au_top_accs_stack_std, alpha=0.1, color="r")
    plt.plot(au_top[0], au_top_accs_stack_mean, '-', color="g", label="Score")

    plt.savefig('all_au_top_n (' + method + ').png')

    prepare_plot("Top Features number (Linear SVM) (" + method + ")", "Top-n AUs",
                 "Score")

    plt.fill_between(fe_top[0], fe_top_accs_stack_mean - fe_top_accs_stack_std, fe_top_accs_stack_mean + fe_top_accs_stack_std, alpha=0.1, color="r")
    plt.plot(fe_top[0], fe_top_accs_stack_mean, '-', color="g", label="Score")

    plt.savefig('all_features_top_n (' + method + ').png')

    prepare_plot("PCA dimension (Linear SVM) (" + method + ")", "Top-n AUs",
                 "Score")

    plt.fill_between(pca_dims[0], pca_dims_accs_stack_mean - pca_dims_accs_stack_std, pca_dims_accs_stack_mean + pca_dims_accs_stack_std, alpha=0.1, color="r")
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
        best_res = res_df[res_df.test_type == '[' + str(test_type) + ']'].sort_values(['mean_test_score'], ascending=False).head(1)
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

        train_idx, test_idx = (data_df[data_df.question_type != test_type].index, data_df[data_df.question_type == test_type].index)

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

        plt.plot(fpr, tpr, color=plt.cm.Paired((2*i + 1) / 12.),
                 label='ROC curve (test type ' + str(i+1) + ', area ' + str(roc_auc) + ')')

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

            _, _, _, subj_mean_fpr, subj_mean_tpr, subj_mean_roc_auc, subj_best_fpr, subj_best_tpr, subj_best_roc_auc =\
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
        
        plt.plot(fpr, tpr, color=plt.cm.tab20(1. * i / 20), label='ROC curve (subject ' + str(i) + ', area ' + str(roc_auc) + ')')
    
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('all_mean_roc_curves (' + method + ').png')

    title = "Best ROC Curves (Linear SVM) (" + method + ")"
    prepare_plot(title, "False Positive Rate", "True Positive Rate")
    
    # plot y=x random guess line
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    
    for i, x in enumerate(zip(best_fprs, best_tprs, best_roc_aucs)):
        fpr, tpr, roc_auc = x
        
        plt.plot(fpr, tpr, color=plt.cm.tab20(1. * i / 20), label='ROC curve (subject ' + str(i) + ', area ' + str(roc_auc) + ')')
    
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

        folds.append((data_df[data_df.question_type != test_type].index, data_df[data_df.question_type == test_type].index))

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

    return train_sizes, train_scores_mean_fin, test_scores_mean_fin, train_scores_std_fin, test_scores_std_fin, best_train_scores, best_test_scores, best_type


def plot_learning_curves_single_subject(raw_path, results_path):
    res_df = pd.read_csv(results_path, index_col=0)

    title = "Learning Curves (Linear SVM) (" + res_df.learning_method.values[0] + ")"

    prepare_plot(title, "Training examples", "Score")

    train_sizes, train_scores_mean_fin, test_scores_mean_fin, train_scores_std_fin, test_scores_std_fin, _, _, _ =\
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

            _, subj_train_scores_mean, subj_test_scores_mean, subj_train_scores_std, subj_test_scores_std, subj_best_train_scores, subj_best_test_scores, subj_best_type =\
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
        
        plt.fill_between(train_sizes, sm - ss, sm + ss, alpha=0.1, color=plt.cm.tab20(1. * (i+1) / 20))
        plt.fill_between(train_sizes, tsm - tss, tsm + tss, alpha=0.1, color=plt.cm.tab20(1. * i / 20))
        
        plt.plot(train_sizes, sm, color=plt.cm.tab20(1. * (i+1) / 20), label="Train (subject {})".format(i))
        plt.plot(train_sizes, tsm, color=plt.cm.tab20(1. * i / 20), label="Test (subject {})".format(i))
    
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('all_mean_learning_curves (' + method + ').png')

    title = "Best Learning Curves (Linear SVM) (" + method + ")"
    prepare_plot(title, "Training examples", "Score")
    plt.grid()

    for i, x in enumerate(zip(best_train_scores, best_test_scores, best_types)):
        b, bt, btype = x
        
        plt.plot(train_sizes, b, color=plt.cm.tab20(1. * (i+1) / 20), label="Train (subject {}, type {})".format(i, btype))
        plt.plot(train_sizes, bt, color=plt.cm.tab20(1. * i / 20), label="Test (subject {}, type {})".format(i, btype))
    
    plt.legend(loc='best', prop={'size': 6})
    plt.savefig('all_best_learning_curves (' + method + ').png')

    title = "Mean of Mean Learning Curves  (Linear SVM) (" + method + ")"
    prepare_plot(title, "Training examples", "Score")
    plt.grid()

    train_scores_mean_stack = np.vstack(train_scores_means).mean(axis=0)
    test_scores_mean_stack = np.vstack(test_scores_means).mean(axis=0)
    train_scores_std_stack = np.vstack(train_scores_stds).mean(axis=0)
    test_scores_std_stack = np.vstack(test_scores_stds).mean(axis=0)

    plt.fill_between(train_sizes, train_scores_mean_stack - train_scores_std_stack, train_scores_mean_stack + train_scores_std_stack, alpha=0.1, color=plt.cm.tab20(1. * (i+1) / 20))
    plt.fill_between(train_sizes, test_scores_mean_stack - test_scores_std_stack, test_scores_mean_stack + test_scores_std_stack, alpha=0.1, color=plt.cm.tab20(1. * i / 20))
    
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
        norm='NO'
):

    features_path = path.join(path.dirname(raw_path), "features__" + features_params_string)
    au_cor_path = path.join(path.dirname(raw_path), "au-correlation__" + features_params_string)
    features_cor_path = path.join(path.dirname(raw_path), "features-correlation__" + features_params_string)

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Choosing Top AU with method", au_selection_method)
    top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)

    if norm != 'NO':
        print("Normalizing with {} method...".format(norm))
        top_AU = utils.normalize_pd_df(top_AU, norm)

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


def cv_method_all_learners(raw_path, ext_features, method, metric=None, features_params_string='', take_sessions=None,
                           timestamp=''):
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
        temp_df['au_method'] = pp_params['au-method']
        temp_df['au_top_n'] = pp_params['au-top-n']
        temp_df['fe_method'] = pp_params['fe-method']
        temp_df['fe_top_n'] = pp_params['fe-top-n']
        temp_df['learning_method'] = pp_params['learning-method']
        temp_df['pca_dim'] = pp_params['pca-dim']
        temp_df['pca_method'] = pp_params['pca-method']
        temp_df['norm'] = pp_params['norm']

        temp_list.append(temp_df.join(pd.DataFrame([method] * len(temp_df), columns=['method'])))

    results_df = pd.concat(temp_list)

    results_path = path.join(path.dirname(raw_path), 'results.{}.{}.csv'.format(pp_params['learning-method'], pp_params['norm']))

    with open(results_path, 'a') as f:
        results_df.to_csv(f, header=False)

    results_df['bete'] = results_df.best_estimator_test_score

    a1 = results_df[results_df.test_type == '[1]']
    a2 = results_df[results_df.test_type == '[2]']
    a3 = results_df[results_df.test_type == '[3]']
    a4 = results_df[results_df.test_type == '[4]']
    a5 = results_df[results_df.test_type == '[5]']

    print(a1.sort_values(['mean_test_score'], ascending=False).head(1).loc[:, ['mean_train_score', 'mean_test_score', 'bete']])
    print(a2.sort_values(['mean_test_score'], ascending=False).head(1).loc[:, ['mean_train_score', 'mean_test_score', 'bete']])
    print(a3.sort_values(['mean_test_score'], ascending=False).head(1).loc[:, ['mean_train_score', 'mean_test_score', 'bete']])
    print(a4.sort_values(['mean_test_score'], ascending=False).head(1).loc[:, ['mean_train_score', 'mean_test_score', 'bete']])
    print(a5.sort_values(['mean_test_score'], ascending=False).head(1).loc[:, ['mean_train_score', 'mean_test_score', 'bete']])

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
    parser.add_argument('-l', '--learning_method', dest='learning_method', type=str,
                        default=SPLIT_METHODS['4_vs_1_all_session_types'],
                        choices=list(SPLIT_METHODS.values()))

    parser.add_argument('-m', '--metric', dest='metric', type=str, default=None)

    # regular, zero-one, one-one, l1, l2, max
    parser.add_argument('-n', '--norm', dest='norm', type=str, default='NO')

    # only take X first sessions for training/test
    parser.add_argument('-t', '--take_sessions', dest='take_sessions', type=int, default=None,
                        choices=list(range(2, 21, 2)))

    # Mega runner: run a lot of different top-au/top-feat/pca-dim combinations
    parser.add_argument('-R', '--mega_runner', dest='mega_runner', action='store_true')

    # plot learning curves, roc curves, and params distribution
    parser.add_argument('-d', '--draw_plots', dest='draw_plots', action='store_true')

    args = parser.parse_args()

    if args.mega_runner:

        timestamp = time.time()

        def mega_run(raw_path, au_selection_method, feature_selection_method, pca_method,
                     learning_method, metric, take_sessions, norm):
            raw_df = pd.read_csv(args.raw_path)

            if norm != 'NO':
                print("Normalizing with {} method...".format(norm))
                raw_df = utils.normalize_pd_df(raw_df, norm)

            for au_top_n in range(51, 40, -1):
                top_au = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)

                for features_top_n in range(18, 32):
                    top_features = utils.get_top_features(top_au, feature_selection_method, features_top_n,
                                                          learning_method, raw_path)

                    if pca_method == PCA_METHODS['global']:
                        pca_options = range(18, 32)
                    else:
                        pca_options = range(3, 6)

                    for pca_dim in pca_options:
                        try:

                            ext_features = utils.dimension_reduction(pca_dim, pca_method, top_features)

                            features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}][pca-dim={}][pca-method={}][learning-method={}][norm={}]'.format(
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

    elif args.draw_plots:
        if args.raw_path is not None and args.results_path is not None:
            plot_learning_curve_single_subject(args.raw_path, args.results_path)
            plot_roc_curve_single_subject(args.raw_path, args.results_path)
            plot_params_dist(args.raw_path, args.results_path)

        else:
            plot_all_learning_curves(args.learning_method)
            plot_all_roc_curves(args.learning_method)
            plot_all_params_dist(args.learning_method)

    else:

        if args.pca_method and args.pca_dim is None:
            parser.error("PCA method (-pm/--pca_method) requires dimension (-p/--pca_dim)")
            exit()


        features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}][pca-dim={}][pca-method={}][learning-method={}][norm={}]'.format(
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
#import path as path
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
