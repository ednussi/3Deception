import itertools
import numpy as np
import sys
from sklearn import cluster as sk_cluster
from constants import RecordFlags
from sklearn.decomposition import PCA
import pandas as pd
from . import moments, discrete_states, dynamic, misc


DROP_COLUMNS = ['question', 'record_flag']
ANSWER_FLAGS = [RecordFlags.RECORD_FLAG_ANSWER_TRUE, RecordFlags.RECORD_FLAG_ANSWER_FALSE]
SKIP_COLUMNS = len(DROP_COLUMNS)


def split_df_to_answers(df):
    """
    Split raw data frame by answers, skip first answer for every question (buffer item)
    """
    answers_df = df[df.record_flag.astype(int).isin(ANSWER_FLAGS)]
    answers_df.index = 'question_' + answers_df.question.astype(str) \
        + '__rid_' + answers_df.record_index.astype(str)
    answers_df = answers_df[answers_df.record_index != 2]

    return [t[1].drop(['question', 'record_index'], axis=1) 
        for t in answers_df.groupby(DROP_COLUMNS)]


def scale(val):
    """
    Scale the given value from the scale of val to the scale of bot-top.
    """
    bot = 0
    top = 1

    if max(val)-min(val) == 0:
        return val
    return ((val - min(val)) / (max(val)-min(val))) * (top-bot) + bot


def quantize(question_dfs, n_quant):
    """
    Quantize values in data frames using K-means
    Args:
        question_dfs: data frames, note that first two columns are not quantized
        n_quant: number of clusters

    Returns:
        list of quantized data frames
    """
    question_quantized_dfs = []

    for q_df in question_dfs:
        q = q_df.copy()
        for au in q.iloc[:, SKIP_COLUMNS:]:
            q.loc[:, au] = sk_cluster\
                .KMeans(n_clusters=n_quant, random_state=1)\
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

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx-delta:
            # if this < delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn+delta:
            # if this > delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

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
    quantized_question_idx_dfs = quantize(question_idx_dfs, n_quant=4)
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


def correlate_features(features_pd,top_features_num):
    # top_features_num - How many features u want
    # return pandas of name of feature and its correlation
    correlation_to_flag = abs(features_pd.corr()['record_flag'])
    correlation_to_flag.sort(ascending=False)
    correlation_to_flag = correlation_to_flag.drop('record_flag')
    top_features = correlation_to_flag[0:top_features_num]
    return top_features


def take_top_features(features_pd,top_features_num):
    top_features = correlate_features(features_pd, top_features_num)
    return features_pd[top_features.index]


"""
To get here for debug reasons:
import os
os.chdir('/cs/engproj/3deception/3deception')
import features
from features import utils, moments, discrete_states, dynamic, misc
import pandas as pd
raw_df = pd.read_csv('output_22-4-17_17-00-00.csv')
question_idx_dfs = utils.split_df_to_questions(raw_df)
"""
