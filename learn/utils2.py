from collections import Counter
from time import time

import scipy
import pandas as pd
from scipy.stats import randint, uniform
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV

from constants import SESSION_TYPES, SPLIT_METHODS, SESSION_TYPE_COLUMN, QUESTION_TYPES, QUESTION_TYPE_COLUMN, \
    SESSION_COLUMN, META_COLUMNS, TARGET_COLUMN, RecordFlags, ANSWER_INDEX_COLUMN, QUESTION_COLUMN


def prepare_classifiers():
    return [
#        {
#            'clf': SGDClassifier(),
#            'params': [
#                {
#                    'n_iter': randint(1, 11),
#                    'alpha': uniform(scale=0.01),
#                    'penalty': ['none', 'l1', 'l2']
#                }
#            ]
#        },
        {
            'clf': svm.SVC(),
            'params': [
                {
                    'C': scipy.stats.expon(scale=100),
                    # 'gamma': scipy.stats.expon(scale=.1),
                    'kernel': ['linear']
                }
            ]
        },
#        {
#            'clf': RandomForestClassifier(),
#            'params': [
#                {
#                    'max_depth': randint(1, 11),
#                    'max_features': randint(1, 8),
#                    'min_samples_split': randint(2, 11),
#                    'min_samples_leaf': randint(1, 11),
#                    'bootstrap': [True, False],
#                    'criterion': ['gini', 'entropy']
#                }
#            ]
#        },
#        {
#            'clf': GradientBoostingClassifier(),
#            'params': [
#                {
#                    'max_depth': randint(1, 11),
#                    'n_estimators': [50, 100, 500, 1000],
#                    'min_samples_split': randint(2, 11),
#                    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
#                    'max_features': ['sqrt', 'log2']
#                }
#            ]
#        }
    ]


def prepare_folds(data_df, method='4v1_A', take_sessions=None):
    """
    Splits features to list of folds 
    according to learning method.
    Each fold is a tuple (train_indices, test_indices)
    
    Supported methods:
        4vs1_A: learn on answers of 4 types of questions and predict every answer in 5th
        4vs1_T/4vs1_F: same as _A, but learn and predict only answers where "Yes"/"No" was said accordingly
        4vs1_SINGLE: same as _A, but learn only on first answer (out of (buffer, first, second))
    
    :param data_df: 
    :param method:
    :param take_sessions:
    :return:
    """

    # print('-- Data preparation for evaluation method: %s' % method)

    session_types = {
        SESSION_TYPES['say_truth']: [],
        SESSION_TYPES['say_lies']: []
    }

    if method.startswith('4v1'):
        if method == SPLIT_METHODS['4_vs_1_all_session_types'] or \
           method == SPLIT_METHODS['4_vs_1_ast_single_answer']:
            session_types[SESSION_TYPES['say_truth']] = [
                RecordFlags.RECORD_FLAG_ANSWER_TRUE,
                RecordFlags.RECORD_FLAG_ANSWER_FALSE
            ]
            session_types[SESSION_TYPES['say_lies']] = [
                RecordFlags.RECORD_FLAG_ANSWER_TRUE,
                RecordFlags.RECORD_FLAG_ANSWER_FALSE
            ]

        elif method == SPLIT_METHODS['4_vs_1_truth_session_types']:
            # Only learn on answers where "YES" was (should have been) answered
            session_types[SESSION_TYPES['say_truth']] = [RecordFlags.RECORD_FLAG_ANSWER_TRUE]
            session_types[SESSION_TYPES['say_lies']] = [RecordFlags.RECORD_FLAG_ANSWER_FALSE]

        elif method == SPLIT_METHODS['4_vs_1_lies_session_types']:
            # Only learn on answers where "NO" was (should have been) answered
            session_types[SESSION_TYPES['say_truth']] = [RecordFlags.RECORD_FLAG_ANSWER_FALSE]
            session_types[SESSION_TYPES['say_lies']] = [RecordFlags.RECORD_FLAG_ANSWER_TRUE]

        else:
            raise Exception('Unknown method: %s' % method)

    # take first n sessions
    if take_sessions is not None:
        data_df = data_df[data_df[SESSION_COLUMN] <= take_sessions]

    # filter answers
    dfs = []

    temp_df = data_df[data_df[SESSION_TYPE_COLUMN].astype(int) == SESSION_TYPES['say_truth']]
    dfs.append(temp_df[temp_df[TARGET_COLUMN].astype(int).isin(session_types[SESSION_TYPES['say_truth']])])

    temp_df = data_df[data_df[SESSION_TYPE_COLUMN].astype(int) == SESSION_TYPES['say_lies']]
    dfs.append(temp_df[temp_df[TARGET_COLUMN].astype(int).isin(session_types[SESSION_TYPES['say_lies']])])

    data_fs = pd.concat(dfs)
    data_fs.reset_index(inplace=True, drop=True)

    if method == SPLIT_METHODS['4_vs_1_ast_single_answer']:
        #  duplicate answer with answer_index=1 to answer_index=2
        ans1_index = data_fs[data_fs[ANSWER_INDEX_COLUMN] == 2].index

        for row in ans1_index:
            q_fs = data_fs[data_fs[QUESTION_COLUMN] == data_fs.loc[row, QUESTION_COLUMN]]
            ans2_row = q_fs[q_fs[ANSWER_INDEX_COLUMN] == 3].index[0]

            data_fs.iloc[ans2_row, len(META_COLUMNS):] = data_fs.iloc[row, len(META_COLUMNS):]
            data_fs.loc[ans2_row, TARGET_COLUMN] = data_fs.loc[row, TARGET_COLUMN]

    folds = []

    loo = LeaveOneOut()

    if method.startswith('4v1'):
        question_types = list(QUESTION_TYPES.values())

        # extract one question type to test on
        for i, (train_val_types_idx, test_types_idx) in enumerate(loo.split(question_types)):

            test_types = list(map(lambda idx: question_types[idx], test_types_idx))

            # extract frames with test question types
            test_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(test_types)].index

            train_val_types = list(map(lambda idx: question_types[idx], train_val_types_idx))

            temp_train_folds, temp_val_folds = [], []
            temp_test_folds = {
                    'indices': test_indices,
                    'types': test_types
                }

            # extract one question type to validate on
            for j, (train_types_idx, val_types_idx) in enumerate(loo.split(train_val_types)):

                train_types = list(map(lambda idx: train_val_types[idx], train_types_idx))
                val_types = list(map(lambda idx: train_val_types[idx], val_types_idx))

                # extract frames with train question types
                train_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(train_types)].index

                # extract frames with validation question types
                val_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(val_types)].index

                temp_train_folds.append({
                    'indices': train_indices,
                    'types': train_types
                })

                temp_val_folds.append({
                    'indices': val_indices,
                    'types': val_types
                })

            # add fold dataset
            folds.append({
                'train': temp_train_folds,
                'val': temp_val_folds,
                'test': temp_test_folds
            })

    elif method == 'SP':
        sessions = data_fs[SESSION_COLUMN].unique()
        session_pairs = list(zip(sessions[::2], sessions[1::2]))

        for i, (train_sessions_idx, test_sessions_idx) in enumerate(loo.split(session_pairs)):

            train_sessions = [s for p in map(lambda idx: session_pairs[idx], train_sessions_idx) for s in p]
            test_sessions = [s for p in map(lambda idx: session_pairs[idx], test_sessions_idx) for s in p]

            # print('   -- Fold {}: train sessions {}, test sessions {}'.format(i, train_sessions, test_sessions))

            # extract frames with train sessions
            train_indices = data_fs[data_fs[SESSION_COLUMN].astype(int).isin(train_sessions)].index

            # extract frames with test sessions
            test_indices = data_fs[data_fs[SESSION_COLUMN].astype(int).isin(test_sessions)].index

            # add data split
            folds.append((train_indices, test_indices))

    else:
        raise Exception('Unknown method: %s' % method)

    return folds, data_fs


def find_params_random_search(clf, param_dist, data, target, folds, score_metric=None):
    """
    Evaluates classifier on given data using score_metric and prepared cross-validation folds
    :param clf: 
    :param param_dist: 
    :param data: 
    :param target: 
    :param folds: 
    :param score_metric: 
    :return: 
    """
    # run randomized search
    n_iter_search = 50
    random_search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring=score_metric,
        cv=folds,
        n_jobs=10
    )
    
    random_search.fit(data, target)
    return random_search


def cv_folds_all_classifiers(data, target, folds, metric=None, method='4v1_A'):
    results = []
    for classifier in prepare_classifiers():
        class_weight = dict(Counter(target))
        class_weight_sum = max(list(class_weight.values()))

        for x in class_weight.keys():
            class_weight[x] = 1. * class_weight[x] / class_weight_sum

        classifier['clf'].set_params(class_weight=class_weight)

        # print('-- Classifier: {}'.format(classifier['clf'].__class__.__name__))
        for param_dict in classifier['params']:

            if method != SPLIT_METHODS['session_prediction']:

                train_val_folds, train_val_types, test_folds, test_types = [], [], [], []

                for f in folds:
                    train_val_folds.append(list(zip(map(lambda x: x['indices'], f['train']),
                                                    map(lambda x: x['indices'], f['val']))))

                    train_val_types.append(list(zip(map(lambda x: x['types'], f['train']),
                                                    map(lambda x: x['types'], f['val']))))

                    test_folds.append(f['test']['indices'])
                    test_types.append(f['test']['types'])

                for i, test_fold in enumerate(test_folds):
                    # concatenate all validation folds to get train data for testing
                    # all validation folds sum is equal to all train folds union
                    train_fold = sum(list(map(lambda x: list(x[1]), train_val_folds[i])), [])

                    train_data = data[train_fold, :]
                    train_target = target[train_fold]

                    test_data = data[test_fold, :]
                    test_target = target[test_fold]

                    clf_cv_result = {
                        'estimator': classifier['clf'].__class__.__name__,
                        'cv_results': find_params_random_search(classifier['clf'],
                                                                param_dict, data, target, train_val_folds[i], metric),
                        'train_types': [q[0] for q in train_val_types[i]],
                        'val_type': [q[1] for q in train_val_types[i]],
                        'test_type': test_types[i]
                    }

                    best_c_param = clf_cv_result['cv_results'].cv_results_['params'][clf_cv_result['cv_results'].best_index_]['C']

                    best_estimator = svm.SVC(kernel='linear', C=best_c_param)
                    best_estimator.fit(train_data, train_target)

                    clf_cv_result['best_mean_val_score'] = clf_cv_result['cv_results'].best_score_
                    clf_cv_result['best_estimator_train_score'] = best_estimator.score(train_data, train_target)
                    clf_cv_result['best_estimator_test_score'] = best_estimator.score(test_data, test_target)
                
                    results.append(clf_cv_result)

            else:
                clf_cv_result = {
                    'estimator': classifier['clf'].__class__.__name__,
                    'cv_results': find_params_random_search(classifier['clf'], param_dict, data, target, folds, metric)
                }

                results.append(clf_cv_result)

    return results


def cv_method_all_classifiers(data_df, method, metric=None, take_sessions=None):
    folds, data_df = prepare_folds(data_df, method, take_sessions)

    data = data_df.iloc[:, len(META_COLUMNS):].values
    target = (data_df[TARGET_COLUMN] == RecordFlags.RECORD_FLAG_ANSWER_TRUE).values

    return cv_folds_all_classifiers(data, target, folds, metric, method)


def cv_all_methods_all_classifiers(data_path, metric=None, take_sessions=None):
    results = []

    for method in SPLIT_METHODS.values():
        # print('Method: {}'.format(method))
        
        results.append({
            'method': method,
            'results': cv_method_all_classifiers(data_path, method, metric, take_sessions)
        })

    return results