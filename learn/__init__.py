from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
import pandas as pd
import numpy as np
from features.utils import SKIP_COLUMNS
from questionnaire.fs_receive import RecordFlags
from time import time
from scipy.stats import randint as sp_randint


def prepare_data(data_path, test_size):
    data_df = pd.read_csv(data_path, index_col=0)

    data = data_df.iloc[:, SKIP_COLUMNS:].values
    target = (data_df.record_flag == RecordFlags.RECORD_FLAG_ANSWER_TRUE).values

    return train_test_split(data, target, test_size=test_size, random_state=42)


def cross_validate_roc_auc(classifier, data, target, folds):
    from sklearn import metrics
    scores = cross_val_score(classifier, data, target, cv=folds, scoring='roc_auc')
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

    return scores


def find_params(clf, param_dist, data, target):
    # run randomized search
    n_iter_search = 20
    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search)

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
