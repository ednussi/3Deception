from time import time

import pandas as pd
from scipy.stats import randint, uniform
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV

from constants import SESSION_TYPES, SPLIT_METHODS, SESSION_TYPE_COLUMN, QUESTION_TYPES, QUESTION_TYPE_COLUMN, \
    SESSION_COLUMN, META_COLUMNS, TARGET_COLUMN, RecordFlags


def prepare_classifiers():
    return [
        {
            'clf': SGDClassifier(),
            'params': [
                {
                    'n_iter': randint(1, 11),
                    'alpha': uniform(scale=0.01),
                    'penalty': ['none', 'l1', 'l2']
                }
            ]
        },
        {
            'clf': svm.SVC(),
            'params': [
                {
                    'C': pd.np.logspace(-2, 5),
                    'kernel': ['linear']
                }
            ]
        },
        {
            'clf': RandomForestClassifier(),
            'params': [
                {
                    'max_depth': randint(1, 11),
                    'max_features': randint(1, 11),
                    'min_samples_split': randint(2, 11),
                    'min_samples_leaf': randint(1, 11),
                    'bootstrap': [True, False],
                    'criterion': ['gini', 'entropy']
                }
            ]
        },
        {
            'clf': GradientBoostingClassifier(),
            'params': [
                {
                    'max_depth': randint(1, 11),
                    'n_estimators': [50, 100, 500, 1000],
                    'min_samples_split': randint(2, 11),
                    'learning_rate': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0],
                    'max_features': ['sqrt', 'log2']
                }
            ]
        }
    ]


def prepare_folds(data_df, method='4v1_A'):
    """
    Splits features to list of folds 
    according to learning method.
    Each fold is a tuple (train_indices, test_indices)
    
    Supported methods:
        4vs1: learn on answers of 4 types of questions and predict every answer in 5th
        sessions: for every session type:
                    learn on n-1 questions in each sessions of this type
                    predict left out questions if it is a part of lying or truthful session
    
    :param data_df: 
    :param method: 
    :return: 
    """

    print('-- Data preparation for evaluation method: %s' % method)

    session_types = [SESSION_TYPES['say_truth'], SESSION_TYPES['say_lies']]

    if method.startswith('4v1'):
        if method == SPLIT_METHODS['4_vs_1_all_session_types']:
            pass

        elif method == SPLIT_METHODS['4_vs_1_truth_session_types']:
            session_types = [SESSION_TYPES['say_truth']]

        elif method == SPLIT_METHODS['4_vs_1_lies_session_types']:
            session_types = [SESSION_TYPES['say_lies']]

        else:
            raise Exception('Unknown method: %s' % method)

    # filter sessions
    data_fs = data_df[data_df[SESSION_TYPE_COLUMN].astype(int).isin(session_types)]

    folds = []

    loo = LeaveOneOut()

    if method.startswith('4v1'):
        question_types = list(QUESTION_TYPES.values())

        for i, (train_types_idx, test_types_idx) in enumerate(loo.split(question_types)):

            train_types = list(map(lambda idx: question_types[idx], train_types_idx))
            test_types = list(map(lambda idx: question_types[idx], test_types_idx))

            print('   -- Fold {}: train question types {}, test question types {}'.format(i, train_types, test_types))

            # extract frames with train question types
            train_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(train_types)].index

            # extract frames with test question types
            test_indices = data_fs[data_fs[QUESTION_TYPE_COLUMN].astype(int).isin(test_types)].index

            # add dataset
            folds.append((train_indices, test_indices))

    elif method == 'SP':
        sessions = data_fs[SESSION_COLUMN].unique()
        session_pairs = list(zip(sessions[::2], sessions[1::2]))

        for i, (train_sessions_idx, test_sessions_idx) in enumerate(loo.split(session_pairs)):

            train_sessions = [s for p in map(lambda idx: session_pairs[idx], train_sessions_idx) for s in p]
            test_sessions = [s for p in map(lambda idx: session_pairs[idx], test_sessions_idx) for s in p]

            print('   -- Fold {}: train sessions {}, test sessions {}'.format(i, train_sessions, test_sessions))

            # extract frames with train sessions
            train_indices = data_fs[data_fs[SESSION_COLUMN].astype(int).isin(train_sessions)].index

            # extract frames with test sessions
            test_indices = data_fs[data_fs[SESSION_COLUMN].astype(int).isin(test_sessions)].index

            # add data split
            folds.append((train_indices, test_indices))

    else:
        raise Exception('Unknown method: %s' % method)

    return folds


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
        n_jobs=2
    )

    random_search.fit(data, target)
    return random_search


def cv_folds_all_classifiers(data, target, folds, metric=None):
    results = []
    for classifier in prepare_classifiers():
        print('-- Classifier: {}'.format(classifier['clf'].__class__.__name__))
        for param_dict in classifier['params']:
            results.append({
                'estimator': classifier['clf'].__class__.__name__,
                'cv_results': find_params_random_search(classifier['clf'], param_dict, data, target, folds, metric)
            })

    return results


def cv_method_all_classifiers(data_path, method, metric=None):
    data_df = pd.read_csv(data_path, index_col=0)

    data = data_df.iloc[:, len(META_COLUMNS):].values
    target = (data_df[TARGET_COLUMN] == RecordFlags.RECORD_FLAG_ANSWER_TRUE).values

    folds = prepare_folds(data_df, method)

    return cv_folds_all_classifiers(data, target, folds, metric)


def cv_all_methods_all_classifiers(data_path, metric=None):
    results = []

    for method in SPLIT_METHODS.values():
        print('Method: {}'.format(method))
        
        results.append({
            'method': method,
            'results': cv_method_all_classifiers(data_path, method, metric)
        })

    return results
