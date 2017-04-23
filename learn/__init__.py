from sklearn.model_selection import train_test_split, cross_val_score
import pandas as pd
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
