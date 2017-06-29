import os.path as path
import itertools
import numpy as np
import sys
from sklearn import cluster as sk_cluster
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
import pandas as pd
from . import moments, discrete_states, dynamic, misc
from constants import *
import pickle


SKIP_COLUMNS = len(META_COLUMNS)
ANSWER_FLAGS = [RecordFlags.RECORD_FLAG_ANSWER_TRUE, RecordFlags.RECORD_FLAG_ANSWER_FALSE]


def split_df_to_answers(df):
    """
    Split raw data frame by questions and answers,
    skip first answer for every question (buffer item)
    """
    answers_df = df[df.record_flag.astype(int).isin(ANSWER_FLAGS)]
    answers_df = answers_df[answers_df.answer_index != 1]

    return [(t[0], t[1]) for t in answers_df.groupby(GROUPBY_COLUMNS)]


def scale(val):
    """
    Scale the given value from the scale of val to the scale of bot-top.
    """
    bot = 0
    top = 1

    if max(val) - min(val) == 0:
        return val
    return ((val - min(val)) / (max(val) - min(val))) * (top - bot) + bot


def quantize(answer_dfs, n_clusters, raw_path=None):
    """
    Quantize values in data frames using K-means
    Args:
        raw_path:
        answer_dfs: data frames, note that first two columns are not quantized
        n_clusters: number of clusters

    Returns:
        list of quantized data frames
    """

    pickle_path = path.join('pickles', '{}.pickle'.format(raw_path.replace('/','_')))

    # check if this raw input was already quantized
    if raw_path is not None and path.isfile(pickle_path):

        # get all quantized answers from pickle
        all_answer_quantized_dfs = pickle.load(open(pickle_path, 'rb'))

    else:  # quantize all answers from raw input

        all_answer_quantized_dfs = []

        raw_df = pd.read_csv(raw_path)

        all_answer_dfs = split_df_to_answers(raw_df)

        for i, q in all_answer_dfs:

            for au in q.iloc[:, SKIP_COLUMNS:]:
                q.loc[:, au] = sk_cluster \
                    .KMeans(n_clusters=n_clusters, random_state=1, n_jobs=10) \
                    .fit_predict(np.reshape(q[au].values, (-1, 1)))

            all_answer_quantized_dfs.append((i, q))

        if raw_path is not None and not path.isfile(pickle_path):
            pickle.dump(all_answer_quantized_dfs, open(pickle_path, 'wb'))

    # get needed answers' ids
    needed_answers = list(map(lambda x: x[0], answer_dfs))

    # leave only needed answers
    answer_quantized_dfs = filter(lambda t: t[0] in needed_answers, all_answer_quantized_dfs)

    answer_quantized_dfs = list(map(lambda t: t[1], answer_quantized_dfs))

    return answer_quantized_dfs


def runs_of_ones_list(bits):
    """
    Calculate lengths of continuous sequences of ones in binary iterable
    Args:
        bits: binary iterable

    Returns:
        array of lengths
    """

    return [sum(g) for b, g in itertools.groupby(bits) if b]


def find_peaks(v, delta=0.1, x=None):
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
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                look_for_max = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                look_for_max = True

    # return np.array(maxtab), np.array(mintab)
    return np.array(maxtab)


def count_peaks(v, delta=0.1, x=None):
    return len(find_peaks(v, delta, x))


def pca_global(df, dim):
    """
    Global dimensionality reduction on all features in df
    """

    # turn data into np array without the ordering columns
    data = df.iloc[:, len(META_COLUMNS):].values.T

    # run PCA
    pca = PCA(n_components=dim, copy=True, whiten=True)
    pca.fit(data)

    # return to DataFrame of proper size
    reduced = pd.concat([df.iloc[:, :len(META_COLUMNS)], pd.DataFrame(pca.components_.T)], axis=1)
    reduced.index = df.index

    return reduced


def pca_grouped(df, groups):
    """
    Local dimensionality reduction per feature group
    groups is a mapping from feature columns to dimension

    Non-meta columns that are not part of any group will not be reduced
    """

    group_columns = set(META_COLUMNS)

    for g in groups.keys():
        group_columns = group_columns.union(set(g))

    skip_columns = set(df.columns) - group_columns

    # turn data into np array without the ordering columns
    reduced = df.copy()
    reduced = reduced.iloc[:, :len(META_COLUMNS)]

    suffix = 0

    for g, dim in groups.items():

        pca = PCA(n_components=dim, copy=True, whiten=True)
        pca.fit(df.loc[:, g].T)

        # append to result DataFrame
        reduced = reduced.join(pd.DataFrame(pca.components_.T,
                                            columns=[str(c)+str(suffix) for c in range(pca.components_.shape[0])]))

        suffix += 1

    # add skipped columns
    reduced = reduced.join(df.loc[:, list(skip_columns)])
    return reduced


def dimension_reduction(pca_dimension, pca_method, top_features):
    if pca_method == PCA_METHODS["global"]:
        # print("Running global PCA...")
        pca_features = pca_global(top_features, pca_dimension)

    elif pca_method == PCA_METHODS["groups"]:
        groups = {
            # moments
            tuple(filter(lambda x: 'mean' in x or
                                   'var' in x or
                                   'skew' in x or
                                   'kurt' in x,
                         top_features.columns)): pca_dimension,

            # dynamic
            tuple(filter(lambda x: 'change_ratio' in x,
                         top_features.columns)): pca_dimension,

            # discreet
            tuple(filter(lambda x: ('change_ratio' not in x and 'ratio' in x) or
                                   '_avg_level' in x or
                                   '_avg_length' in x or
                                   'avg_volume' in x,
                         top_features.columns)): pca_dimension,
        }

        # print("Running PCA for feature groups...")
        pca_features = pca_grouped(top_features, groups)

    else:
        raise Exception("Unknown PCA method")

    return pca_features


def get_all_features_by_groups(raw_df, raw_path=None):
    print("Splitting answers... ", end="")
    answers_dfs = split_df_to_answers(raw_df)
    print("Done.")

    print("Quantizing... ", end="")
    quantized_answers_dfs = quantize(answers_dfs, n_clusters=4, raw_path=raw_path)
    print("Done.")

    # remove unneeded indices
    answers_dfs = list(map(lambda t: t[1], answers_dfs))

    print("Moments... ", end="")
    all_moments = moments.moments(answers_dfs)
    print("Done.")

    print("Discrete... ", end="")
    all_discrete = discrete_states.discrete_states(quantized_answers_dfs)
    print("Done.")

    print("Dynamic... ", end="")
    all_dynamic = dynamic.dynamic(quantized_answers_dfs)
    print("Done.")

    print("Miscellaneous... ", end="")
    all_misc = misc.misc(answers_dfs)
    print("Done.")

    return all_moments, all_discrete, all_dynamic, all_misc


def get_all_features(raw_df, raw_path=None):
    all_moments, all_discrete, all_dynamic, all_misc = get_all_features_by_groups(raw_df, raw_path)

    all_features = pd.concat([
        all_moments,
        all_discrete,
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


def get_corr_(df, top_n, method):
    if method == 'SP':
        label_col = 'session_type'
    else:
        label_col = 'record_flag'

    meta = [x for x in META_COLUMNS]

    meta.remove(label_col)

    if method.startswith('SP'):  # in case of session prediction, we need record flag to filter only answer frames
        meta.remove(TARGET_COLUMN)

    data = df.drop(meta, axis=1)

    correlation_to_flag = abs(data[data.record_flag.astype(int).isin(ANSWER_FLAGS)].corr()[label_col])
    correlation_to_flag = correlation_to_flag.sort_values(ascending=False)
    correlation_to_flag = correlation_to_flag.drop([label_col])

    if method.startswith('SP'):  # drop record flag now
        correlation_to_flag = correlation_to_flag.drop([TARGET_COLUMN])

    return correlation_to_flag[0:top_n]


def take_top_(df, top_n, method):
    # top_features_num - How many features u want
    # return pandas of name of feature and its correlation
    identifiers = df[META_COLUMNS]
    top_features = get_corr_(df, top_n, method)
    top_features_pd = identifiers.join(df[top_features.keys()])
    return top_features_pd


def normalize_pd_df(df, norm_type):
    data_df = df.iloc[:, len(META_COLUMNS):]
    identifiers = df[META_COLUMNS]

    if norm_type == 'regular':
        # Regular normalization
        df_norm = ((data_df - data_df.mean()) / data_df.std()).fillna(0.0)

    elif norm_type == 'zero-one':
        # Normalize Features to [0,1]
        df_norm = ((data_df - data_df.min()) / (data_df.max() - data_df.min())).fillna(0.0)

    elif norm_type == 'one-one':
        # Normalize Features to [-1,1]
        df_norm = (2 * (data_df - data_df.min()) / (data_df.max() - data_df.min()) - 1).fillna(0.0)

    elif norm_type == 'l1' or norm_type == 'l2' or norm_type == 'max':
        df_norm = normalize(data_df, norm=norm_type, axis=0)
        df_norm = pd.DataFrame(df_norm, columns=data_df.columns, index=data_df.index)
    else:
        raise Exception('Unknown regularization type')

    return identifiers.join(df_norm)


def get_top_au(raw_df, au, au_num, method):
    if au == 'daniel':
        return take_top_(raw_df[META_COLUMNS].join(raw_df[GOOD_DANIEL_AU]), au_num, method)
        
    elif au == 'mouth':
        return take_top_(raw_df[META_COLUMNS].join(raw_df[MOUTH_AU]), au_num, method)
        
    elif au == 'eyes':
        return take_top_(raw_df[META_COLUMNS].join(raw_df[EYES_AREA_AU]), au_num, method)
        
    elif au == 'eyes_area':
        return take_top_(raw_df[META_COLUMNS].join(raw_df[EYES_AU]), au_num, method)
        
    elif au == 'brows':
        return take_top_(raw_df[META_COLUMNS].join(raw_df[BROWS_AU]), au_num, method)
        
    elif au == 'smile':
        return take_top_(raw_df[META_COLUMNS].join(raw_df[SMILE_AU]), au_num, method)
        
    elif au == 'blinks':
        return take_top_(raw_df[META_COLUMNS].join(raw_df[BLINKS_AU]), au_num, method)
        
    else:  # elif au == 'top':
        return take_top_(raw_df, au_num, method)

def get_top_by_method(raw_df,au, au_num, method):
    if au == 'daniel':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[GOOD_DANIEL_AU])

    elif au == 'mouth':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[MOUTH_AU])

    elif au == 'eyes':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[EYES_AREA_AU])

    elif au == 'eyes_area':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[EYES_AU])

    elif au == 'brows':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[BROWS_AU])

    elif au == 'smile':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[SMILE_AU])

    elif au == 'blinks':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[BLINKS_AU])

    else:  # elif au == 'top':
        raw_df_au = raw_df

    # top_features_num - How many features u want
    # return pandas of name of feature and its correlation
    return get_corr_(raw_df_au, au_num, method)


def get_top_au2(raw_df, test_df, au, au_num, method):
    if au == 'daniel':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[GOOD_DANIEL_AU])

    elif au == 'mouth':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[MOUTH_AU])

    elif au == 'eyes':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[EYES_AREA_AU])

    elif au == 'eyes_area':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[EYES_AU])

    elif au == 'brows':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[BROWS_AU])

    elif au == 'smile':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[SMILE_AU])

    elif au == 'blinks':
        raw_df_au = raw_df[META_COLUMNS].join(raw_df[BLINKS_AU])

    else:  # elif au == 'top':
        raw_df_au = raw_df

    top_features = get_top_by_method(raw_df,au, au_num, method)
    identifiers = raw_df_au[META_COLUMNS]
    top_features_pd = identifiers.join(raw_df_au[top_features.keys()])
    test_identifiers = test_df[META_COLUMNS]
    test_top_features_pd = test_identifiers.join(test_df[top_features.keys()])

    return top_features_pd, test_top_features_pd

def partition(lst):
    n = 4  # number of groups
    division = len(lst) / n
    return [len(lst[round(division * i):round(division * (i + 1))]) for i in range(n)]


def get_top_features(top_AU, feature_selection_method, features_top_n, learning_method, raw_path=None):
    if feature_selection_method == 'all':
        all_features = get_all_features(top_AU, raw_path)
        return take_top_(all_features, features_top_n, learning_method).fillna(0.0)

    else:  # elif feat == 'by group'
        au_per_group = partition(list(range(features_top_n)))
        all_list = list(get_all_features_by_groups(top_AU, raw_path))  # all_moments, all_discrete, all_dynamic, all_misc
        
        top_feature_group_list = [None] * 4
        
        for i in range(1,4):
            all_list[i] = all_list[0].loc[:,META_COLUMNS].join(all_list[i])


        for i in range(4):
            top_feature_group_list[i] = take_top_(all_list[i], au_per_group[i], learning_method)

        return pd.concat([
            top_feature_group_list[0],
            top_feature_group_list[1].iloc[:, len(META_COLUMNS):],
            top_feature_group_list[2].iloc[:, len(META_COLUMNS):],
            top_feature_group_list[3].iloc[:, len(META_COLUMNS):]
        ], axis=1).fillna(0.0)


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
