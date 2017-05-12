import itertools
import numpy as np
import sys
from sklearn import cluster as sk_cluster
from constants import RecordFlags
from sklearn.decomposition import PCA
import pandas as pd
from . import moments, discrete_states, dynamic, misc


GOOD_DANIEL_AU = ['EyeBlink_L', 'EyeBlink_R','EyeIn_L', 'EyeIn_R', 'BrowsU_C', 'BrowsU_L', 'BrowsU_R', 'JawOpen', 'MouthLeft',
                    'MouthRight', 'MouthFrown_L', 'MouthFrown_R', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L',
                    'MouthDimple_R', 'LipsStretch_L', 'LipsStretch_R', 'LipsUpperUp', 'LipsFunnel', 'ChinLowerRaise',
                    'Sneer', 'CheekSquint_L', 'CheekSquint_R']
META_COLUMNS = ["session", "question", "question_type", "record_flag", "answer_index", "timestamp"]
GROUPBY_COLUMNS = ["question", "answer_index"]
ANSWER_FLAGS = [RecordFlags.RECORD_FLAG_ANSWER_TRUE, RecordFlags.RECORD_FLAG_ANSWER_FALSE]


def split_df_to_answers(df):
    """
    Split raw data frame by questions and answers, 
    skip first answer for every question (buffer item)
    """
    answers_df = df[df.record_flag.astype(int).isin(ANSWER_FLAGS)]
    answers_df = answers_df[answers_df.answer_index != 2]

    answers_df.index = ['__'.join(ind) for ind in zip(*[[x + '_'] * len(answers_df[x]) + answers_df[x].astype(str) for x in META_COLUMNS])]

    return [t[1].drop(META_COLUMNS, axis=1) for t in answers_df.groupby(GROUPBY_COLUMNS)]


def cv_split_df_to_answers(df):
    """
    Split raw data frame by questions and answers, 
    skip first answer for every question (buffer item).
    
    Split 5 existing questions to cross-validation dataset:
     3 questions for training, 1 for validation, 1 for test
    """
    for test_question_number in df.question_type.unique():

        if test_question_number == 0:
            continue

        train_validation_questions = df[df.question_type != test_question_number]

        test_questions = df[df.question_type == test_question_number]

        for validation_question_number in train_validation_questions.question_type.unique():

            if validation_question_number == 0:
                continue

            train_questions = \
                train_validation_questions[train_validation_questions.question_type != validation_question_number]

            validation_questions = \
                train_validation_questions[train_validation_questions.question_type == validation_question_number]

            train = split_df_to_answers(train_questions)
            validation = split_df_to_answers(validation_questions)
            test = split_df_to_answers(test_questions)

            yield (train, validation, test)


def scale(val):
    """
    Scale the given value from the scale of val to the scale of bot-top.
    """
    bot = 0
    top = 1

    if max(val)-min(val) == 0:
        return val
    return ((val - min(val)) / (max(val)-min(val))) * (top-bot) + bot


def quantize(question_dfs, n_clusters):
    """
    Quantize values in data frames using K-means
    Args:
        question_dfs: data frames, note that first two columns are not quantized
        n_clusters: number of clusters

    Returns:
        list of quantized data frames
    """
    question_quantized_dfs = []

    for q_df in question_dfs:
        q = q_df.copy()
        for au in q.iloc[:, SKIP_COLUMNS:]:
            q.loc[:, au] = sk_cluster\
                .KMeans(n_clusters=n_clusters, random_state=1)\
                .fit_predict(np.reshape(q[au].values, (-1, 1)))

        question_quantized_dfs.append(q)

    return question_quantized_dfs


def runs_of_ones_list(bits):
    """
    Calculate lengths of continuous sequences of ones in binary iterable
    Args:
        bits: binary iterable

    Returns:
        array of lengths
    """

    return [sum(g) for b, g in itertools.groupby(bits) if b]


def find_peaks(v, delta=0.1, x = None):
    """
    Converted from MATLAB script at http://billauer.co.il/peakdet.html
    Returns two arrays
    function [maxtab, mintab]=peakdet(v, delta, x)
    %PEAKDET Detect peaks in a vector
    %        [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    %        maxima and minima ("peaks") in the vector V.
    %        MAXTAB and MINTAB consists of two columns. Column 1
    %        contains indices in V, and column 2 the found values.
    %
    %        With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    %        in MAXTAB and MINTAB are replaced with the corresponding
    %        X-values.
    %
    %        A point is considered a maximum peak if it has the maximal
    %        value, and was preceded (to the left) by a value lower by
    %        DELTA.
    % Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    % This function is released to the public domain; Any use is allowed.
    """
    maxtab = []
    mintab = []

    if x is None:
        x = np.arange(len(v))

    v = np.asarray(v)

    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta < 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    look_for_max = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if look_for_max:
            if this < mx-delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                look_for_max = False
        else:
            if this > mn+delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                look_for_max = True

    # return np.array(maxtab), np.array(mintab)
    return np.array(maxtab)


def count_peaks(v, delta=0.1, x = None):
    return len(find_peaks(v, delta, x))


def pca_3d(df, dim):

    # turn data into np array without the ordering columns
    data = df.iloc[:, 3:].values.T

    # run PCA
    pca = PCA(n_components=dim, copy=True, whiten=True)
    pca.fit(data)

    # return to DataFrame of proper size
    reduced = pd.concat([df.iloc[:, :3], pd.DataFrame(pca.components_.T)], axis=1)
    reduced.index = df.index

    return reduced


def get_all_features(raw_df):
    print("Splitting answers... ", end="")
    question_idx_dfs = split_df_to_answers(raw_df)
    print("Done.")

    print("Quantizing... ", end="")
    quantized_question_idx_dfs = quantize(question_idx_dfs, n_clusters=4)
    print("Done.")

    print("Moments... ", end="")
    all_moments = moments.moments(question_idx_dfs)
    print("Done.")

    print("Discrete... ", end="")
    all_discrete = discrete_states.discrete_states(quantized_question_idx_dfs)
    print("Done.")

    print("Dynamic... ", end="")
    all_dynamic = dynamic.dynamic(quantized_question_idx_dfs)
    print("Done.")

    print("Miscellaneous... ", end="")
    all_misc = misc.misc(question_idx_dfs)
    print("Done.")

    all_features = pd.concat([
        all_moments,
        all_discrete.iloc[:, SKIP_COLUMNS:],
        all_dynamic,
        all_misc
    ], axis=1)

    all_features.index = all_moments.index

    return all_features


def select_features_from_model(clf, X, y):
    from sklearn.feature_selection import SelectFromModel
    clf = clf.fit(X, y)

    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)

    return X_new


def extract_select_tsflesh_features(X):
    from tsfresh import extract_relevant_features

    y = X.record_flag

    features = X
    features['id'] = features.index
    features.drop(['record_flag'], axis=1)

    features_filtered_direct = extract_relevant_features(
        features, y, column_id='id', column_sort='timestamp')

    return features_filtered_direct

def take_top_features(features_pd,top_features_num):
    # top_features_num - How many features u want
    # return pandas of name of feature and its correlation
    print(features_pd.shape)
    identifiers = features_pd.iloc[:,:6]
    print(identifiers.shape)
    data = features_pd.iloc[:,[3]+list(range(7,len(features_pd.columns)))] #column 3 is record_flag
    print(data.shape)
    correlation_to_flag = abs(data.corr()['record_flag'])
    correlation_to_flag.sort(ascending=False)
    correlation_to_flag = correlation_to_flag.drop('record_flag')
    top_features = correlation_to_flag[0:top_features_num]
    top_features_pd = identifiers.join(features_pd[top_features.keys()])
    print(top_features_pd.shape)
    return top_features_pd

def normalize_pd_df(df):
    # Regular normalization
    df_norm = (df - df.mean()) / df.std()
    # Normalize Features to [0,1]
    # df_norm = (df - df.min()) / (df.max() - df.min())
    # Normalize Features to [-1,1]
    # df_norm = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_norm

def get_top_au(raw_df, method, num_au):
    if method == 'daniel':
        return raw_df[META_COLUMNS].join(raw_df[GOOD_DANIEL_AU])
    elif method == 'top':
        return take_top_features(raw_df, num_au)
    else: #elif method == 'all':
        return raw_df



"""
To get here for debug reasons:
import os
os.chdir('C:/Users/owner/Desktop/Project/3deception/')
os.chdir('/cs/engproj/3deception/3deception')
import features
from features import utils, moments, discrete_states, dynamic, misc
import pandas as pd
raw_df = pd.read_csv('output_22-4-17_17-00-00.csv')
all_features = utils.get_all_features(raw_df)
"""
