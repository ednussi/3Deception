import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from constants import RecordFlags, META_COLUMNS, TARGET_COLUMN
from time import time


def prepare_data(data_path, method='4v1_A'):
    """
    Reads features from csv file, and splits to list of datasets 
    according to learning method.
    Each dataset is a tuple (train, test, train_labels, test_labels)
    
    Supported methods:
        4vs1: learn on answers of 4 types of questions and predict every answer in 5th
        sessions: for every session type:
                    learn on n-1 questions in each sessions of this type
                    predict left out questions if it is a part of lying or truthful session
    
    :param data_path: 
    :param method: 
    :return: 
    """
    from constants import SESSION_COLUMN, SESSION_TYPE_COLUMN, QUESTION_TYPE_COLUMN, SESSION_TYPES, QUESTION_TYPES
    from sklearn.model_selection import LeaveOneOut

    print('Data preparation for evaluation method: %s' % method)

    # read input features
    data_df = pd.read_csv(data_path)

    session_types = [SESSION_TYPES['say_truth'], SESSION_TYPES['say_lies']]

    if method.startswith('4v1'):
        if method.endswith('A'):
            pass

        elif method.endswith('T'):
            session_types = [SESSION_TYPES['say_truth']]

        elif method.endswith('F'):
            session_types = [SESSION_TYPES['say_lies']]

        else:
            raise Exception('Unknown method: %s' % method)

    # filter sessions
    data_df = data_df[data_df[SESSION_TYPE_COLUMN].astype(int).isin(session_types)]

    data_sets = []

    loo = LeaveOneOut()

    if method.startswith('4v1'):
        question_types = list(QUESTION_TYPES.values())

        for i, (train_types_idx, test_types_idx) in enumerate(loo.split(question_types)):

            train_types = list(map(lambda idx: question_types[idx], train_types_idx))
            test_types = list(map(lambda idx: question_types[idx], test_types_idx))

            print('-- Fold {}: train question types {}, test question types {}'.format(i, train_types, test_types))

            # extract frames with train question types
            train = data_df[data_df[QUESTION_TYPE_COLUMN].astype(int).isin(train_types)]

            # drop meta columns
            train_data = train.iloc[:, len(META_COLUMNS):].values

            # extract train labels
            train_labels = (train[TARGET_COLUMN] == RecordFlags.RECORD_FLAG_ANSWER_TRUE).values

            # extract frames with test question types
            test = data_df[data_df[QUESTION_TYPE_COLUMN].astype(int) == test_types[0]]

            # drop meta columns
            test_data = test.iloc[:, len(META_COLUMNS):].values

            # extract test labels
            test_labels = (test[TARGET_COLUMN] == RecordFlags.RECORD_FLAG_ANSWER_TRUE).values

            # add dataset
            data_sets.append((train_data, test_data, train_labels, test_labels))

    elif method == 'SP':
        sessions = data_df[SESSION_COLUMN].unique()
        session_pairs = list(zip(sessions[::2], sessions[1::2]))

        for i, (train_sessions_idx, test_sessions_idx) in enumerate(loo.split(session_pairs)):

            train_sessions = [s for p in map(lambda idx: session_pairs[idx], train_sessions_idx) for s in p]
            test_sessions = [s for p in map(lambda idx: session_pairs[idx], test_sessions_idx) for s in p]

            print('-- Fold {}: train sessions {}, test sessions {}'.format(i, train_sessions, test_sessions))

            # extract frames with train sessions
            train = data_df[data_df[SESSION_COLUMN].astype(int).isin(train_sessions)]

            # drop meta columns
            train_data = train.iloc[:, len(META_COLUMNS):].values

            # extract train labels
            train_labels = (train[SESSION_TYPE_COLUMN] == SESSION_TYPES['say_truth']).values

            # extract frames with test sessions
            test = data_df[data_df[SESSION_COLUMN].astype(int).isin(test_sessions)]

            # drop meta columns
            test_data = test.iloc[:, len(META_COLUMNS):].values

            # extract train labels
            test_labels = (test[SESSION_TYPE_COLUMN] == SESSION_TYPES['say_truth']).values

            # add dataset
            data_sets.append((train_data, test_data, train_labels, test_labels))

    else:
        raise Exception('Unknown method: %s' % method)

    return data_sets


def find_params_random_search(clf, param_dist, data, target, score_metric):
    # run randomized search
    n_iter_search = 50
    random_search = RandomizedSearchCV(
        clf,
        param_distributions=param_dist,
        n_iter=n_iter_search,
        scoring=score_metric
    )

    start = time()
    random_search.fit(data, target)
    print("RandomizedSearchCV took %.2f seconds for %d candidates parameter settings."
          % ((time() - start), n_iter_search))

    report(random_search.cv_results_)


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                results['mean_test_score'][candidate],
                results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")
