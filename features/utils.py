import os.path as path
import itertools
import numpy as np
import sys
from sklearn import cluster as sk_cluster
from constants import RecordFlags
from sklearn.decomposition import PCA
import pandas as pd
from . import moments, discrete_states, dynamic, misc
from constants import META_COLUMNS, GROUPBY_COLUMNS

SKIP_COLUMNS = len(META_COLUMNS)
ANSWER_FLAGS = [RecordFlags.RECORD_FLAG_ANSWER_TRUE, RecordFlags.RECORD_FLAG_ANSWER_FALSE]

ALL_AU = ['EyeBlink_L', 'EyeBlink_R', 'EyeSquint_L', 'EyeSquint_R', 'EyeDown_L', 'EyeDown_R', 'EyeIn_L', 'EyeIn_R',
          'EyeOpen_L',
          'EyeOpen_R', 'EyeOut_L', 'EyeOut_R', 'EyeUp_L', 'EyeUp_R', 'BrowsD_L', 'BrowsD_R', 'BrowsU_C', 'BrowsU_L',
          'BrowsU_R',
          'JawFwd', 'JawLeft', 'JawOpen', 'JawChew', 'JawRight', 'MouthLeft', 'MouthRight', 'MouthFrown_L',
          'MouthFrown_R',
          'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L', 'MouthDimple_R', 'LipsStretch_L', 'LipsStretch_R',
          'LipsUpperClose',
          'LipsLowerClose', 'LipsUpperUp', 'LipsLowerDown', 'LipsUpperOpen', 'LipsLowerOpen', 'LipsFunnel',
          'LipsPucker', 'ChinLowerRaise',
          'ChinUpperRaise', 'Sneer', 'Puff', 'CheekSquint_L', 'CheekSquint_R']

ALL_AU_DANIEL = ['EyeBlink_L', 'EyeBlink_R', 'EyeSquint_L', 'EyeSquint_R', 'EyeDown_L', 'EyeDown_R', 'EyeIn_L',
                 'EyeIn_R',
                 'EyeOpen_L', 'EyeOpen_R', 'EyeOut_L', 'EyeOut_R', 'EyeUp_L', 'EyeUp_R', 'BrowsD_L', 'BrowsD_R',
                 'BrowsU_C', 'BrowsU_L', 'BrowsU_R', 'JawOpen', 'LipsTogether', 'JawLeft', 'JawRight', 'JawFwd',
                 'LipsUpperUp_L', 'LipsUpperUp_R', 'LipsLowerDown_L', 'LipsLowerDown_R', 'LipsUpperClose',
                 'LipsLowerClose', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L', 'MouthDimple_R', 'LipsStretch_L',
                 'LipsStretch_R', 'MouthFrown_L', 'MouthFrown_R', 'MouthPress_L', 'MouthPress_R', 'LipsPucker',
                 'LipsFunnel', 'MouthLeft', 'MouthRight', 'ChinLowerRaise', 'ChinUpperRaise', 'Sneer_L', 'Sneer_R',
                 'Puff', 'CheekSquint_L', 'CheekSquint_R']  # len = 51

GOOD_DANIEL_AU = ['EyeBlink_L', 'EyeBlink_R', 'EyeIn_L', 'EyeIn_R', 'BrowsU_C', 'BrowsU_L', 'BrowsU_R', 'JawOpen',
                  'MouthLeft',
                  'MouthRight', 'MouthFrown_L', 'MouthFrown_R', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L',
                  'MouthDimple_R', 'LipsStretch_L', 'LipsStretch_R', 'LipsUpperUp', 'LipsFunnel', 'ChinLowerRaise',
                  'Sneer', 'CheekSquint_L', 'CheekSquint_R']

MOUTH_AU = ['JawFwd', 'JawLeft', 'JawOpen', 'JawChew', 'JawRight', 'LipsUpperUp', 'LipsLowerDown', 'LipsUpperClose',
            'LipsUpperOpen', 'LipsLowerOpen', 'LipsLowerClose', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L',
            'LipsUpperOpen', 'LipsLowerOpen', 'LipsLowerClose', 'MouthSmile_L', 'MouthSmile_R', 'MouthDimple_L',
            'MouthDimple_R', 'LipsStretch_L', 'LipsStretch_R', 'MouthFrown_L', 'MouthFrown_R', 'LipsPucker',
            'LipsFunnel', 'MouthLeft', 'MouthRight', 'ChinLowerRaise', 'ChinUpperRaise']

EYES_AREA_AU = ['EyeBlink_L', 'EyeBlink_R', 'EyeSquint_L', 'EyeSquint_R', 'EyeDown_L', 'EyeDown_R', 'EyeIn_L',
                'EyeIn_R', 'EyeOpen_L',
                'EyeOpen_R', 'EyeOut_L', 'EyeOut_R', 'EyeUp_L', 'EyeUp_R', 'BrowsD_L', 'BrowsD_R',
                'BrowsU_C', 'BrowsU_L', 'BrowsU_R']

EYES_AU = ['EyeBlink_L', 'EyeBlink_R', 'EyeSquint_L', 'EyeSquint_R', 'EyeDown_L', 'EyeDown_R', 'EyeIn_L', 'EyeIn_R',
           'EyeOpen_L',
           'EyeOpen_R', 'EyeOut_L', 'EyeOut_R', 'EyeUp_L', 'EyeUp_R']

BROWS_AU = ['BrowsD_L', 'BrowsD_R', 'BrowsU_C', 'BrowsU_L', 'BrowsU_R']
SMILE_AU = ['MouthSmile_L', 'MouthSmile_R']
BLINKS_AU = ['EyeBlink_L', 'EyeBlink_R']


def split_df_to_answers(df):
    """
    Split raw data frame by questions and answers, 
    skip first answer for every question (buffer item)
    """
    answers_df = df[df.record_flag.astype(int).isin(ANSWER_FLAGS)]
    answers_df = answers_df[answers_df.answer_index != 2]

    return [t[1] for t in answers_df.groupby(GROUPBY_COLUMNS)]


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

    if max(val) - min(val) == 0:
        return val
    return ((val - min(val)) / (max(val) - min(val))) * (top - bot) + bot


def quantize(question_dfs, n_clusters, raw_path=None):
    """
    Quantize values in data frames using K-means
    Args:
        raw_path:
        question_dfs: data frames, note that first two columns are not quantized
        n_clusters: number of clusters

    Returns:
        list of quantized data frames
    """
    question_quantized_dfs = []

    for i, q_df in enumerate(question_dfs):
        pickle_path = 'pickles/{}__quantized_answers_df_{}.pickle'.format(raw_path, i)
        if raw_path is not None and path.isfile(pickle_path):
            q = pd.read_pickle(pickle_path)

        else:
            q = q_df.copy()
            for au in q.iloc[:, SKIP_COLUMNS:]:
                q.loc[:, au] = sk_cluster \
                    .KMeans(n_clusters=n_clusters, random_state=1) \
                    .fit_predict(np.reshape(q[au].values, (-1, 1)))

            if raw_path is not None and not path.isfile(pickle_path):
                q.to_pickle(pickle_path)

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
        group_columns.union(set(g))

    skip_columns = set(df.columns) - group_columns

    # turn data into np array without the ordering columns
    reduced = df.copy()
    reduced = reduced.iloc[:, :len(META_COLUMNS)]

    for g, dim in groups.items():

        pca = PCA(n_components=dim, copy=True, whiten=True)
        pca.fit(df.loc[:, g])

        # append to result DataFrame
        reduced = pd.concat([reduced, pd.DataFrame(pca.components_.T, index=reduced.index)], axis=1)

    # add skipped columns
    reduced = pd.concat([reduced, df.loc[:, list(skip_columns)]])
    return reduced


def get_all_features_by_groups(raw_df, raw_path=None):
    print("Splitting answers... ", end="")
    answers_dfs = split_df_to_answers(raw_df)
    print("Done.")

    print("Quantizing... ", end="")
    quantized_answers_dfs = quantize(answers_dfs, n_clusters=4, raw_path=raw_path)
    print("Done.")

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


def take_top_(df, top_n, method):
    # top_features_num - How many features u want
    # return pandas of name of feature and its correlation
    meta = [x for x in META_COLUMNS]
    identifiers = df[META_COLUMNS]
    if method == 'SP':
        label_col = 'session_type'
    else:
        label_col = 'record_flag'

    meta.remove(label_col)
    data = df.drop(meta, axis=1)
    correlation_to_flag = abs(data.corr()[label_col])
    correlation_to_flag.sort(ascending=False)
    correlation_to_flag = correlation_to_flag.drop(label_col)
    top_features = correlation_to_flag[0:top_n]
    top_features_pd = identifiers.join(df[top_features.keys()])
    return top_features_pd


def normalize_pd_df(df):
    # Regular normalization
    df_norm = (df - df.mean()) / df.std()
    # Normalize Features to [0,1]
    # df_norm = (df - df.min()) / (df.max() - df.min())
    # Normalize Features to [-1,1]
    # df_norm = 2 * (df - df.min()) / (df.max() - df.min()) - 1
    return df_norm


def get_top_au(raw_df, au, au_num, method):
    if au == 'daniel':
        return raw_df[META_COLUMNS].join(raw_df[GOOD_DANIEL_AU])
    elif au == 'mouth':
        return raw_df[META_COLUMNS].join(raw_df[MOUTH_AU])
    elif au == 'eyes':
        return raw_df[META_COLUMNS].join(raw_df[EYES_AREA_AU])
    elif au == 'eyes_area':
        return raw_df[META_COLUMNS].join(raw_df[EYES_AU])
    elif au == 'brows':
        return raw_df[META_COLUMNS].join(raw_df[BROWS_AU])
    elif au == 'smile':
        return raw_df[META_COLUMNS].join(raw_df[SMILE_AU])
    elif au == 'blinks':
        return raw_df[META_COLUMNS].join(raw_df[BLINKS_AU])
    else:  # elif au == 'top':
        return take_top_(raw_df, au_num, method)


def partition(lst):
    n = 4  # number of groups
    division = len(lst) / n
    return [len(lst[round(division * i):round(division * (i + 1))]) for i in range(n)]


def get_top_features(top_AU, feat, feat_num, method, raw_path=None):
    if feat == 'all':
        all_features = get_all_features(top_AU, raw_path)
        return take_top_(all_features, feat_num, method)

    else:  # elif feat == 'by group'
        au_per_group = partition(list(range(feat_num)))
        all_list = list(get_all_features_by_groups(top_AU, raw_path))  # all_moments, all_discrete, all_dynamic, all_misc
        for i in range(len(all_list)):
            all_list[i] = pd.concat([top_AU[META_COLUMNS],all_list[i]])

        top_feature_group_list = [None] * 4
        for i in range(4):
            top_feature_group_list[i] = take_top_(all_list[i], au_per_group[i], method)

        return pd.concat([
            top_feature_group_list[0],
            top_feature_group_list[1].iloc[:, len(META_COLUMNS):],
            top_feature_group_list[2].iloc[:, len(META_COLUMNS):],
            top_feature_group_list[3].iloc[:, len(META_COLUMNS):]
        ], axis=1)


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
