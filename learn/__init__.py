from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
import numpy as np
from features.utils import SKIP_COLUMNS
from questionnaire.fs_receive import RecordFlags


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


def train_svm(classifier, data_path):
    train, test, train_labels, test_labels = prepare_data(data_path, 0.25)

    classifier.fit(train, train_labels)
    predicted = classifier.predict(test)

    accuracy = 1. * sum(abs(predicted.astype(int) - test_labels.astype(int))) / len(test_labels)
    print('Accuracy: {}'.format(accuracy))


def try_svms():
    train, test, train_labels, test_labels = prepare_data(data_path, 0.25)
    n_features = train.shape[1]



    for kernel in ['linear', 'poly', 'rdb', 'sigmoid', 'precomputed']:
        if kernel == 'poly':
            for degree in range(1,10):

        if kernel == 'poly' or kernel == 'rbf' or kernel == 'sigmoid':
            for gamma in np.linspace(n_features,1,10):

        if 


