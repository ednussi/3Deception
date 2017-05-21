import argparse
import numpy as np
import pandas as pd
import os.path as path
from features import utils
from learn.utils import cv_method_all_classifiers
from constants import PCA_METHODS, SPLIT_METHODS, AU_SELECTION_METHODS, FEATURE_SELECTION_METHODS


def cv_method_all_learners(raw_path, pca_features, method, metric=None, features_params_string='', take_sessions=None):
    # print("Cross validating all learners...")
    results = cv_method_all_classifiers(pca_features, method, metric, take_sessions)

    temp_list = []
    
    for x in results:
        temp_df = pd.DataFrame(x['cv_results'].cv_results_)
        temp_df['real_test_score'] = x['best_estimator_test_score']
        temp_df['train_types'] = x['train_types']
        temp_df['val_type'] = x['val_type']
        temp_df['test_type'] = x['test_type']
        temp_list.append(temp_df.join(pd.DataFrame([method] * len(temp_df), columns=['method'])))

    results_df = pd.concat(temp_list)

    results_path = path.join(path.dirname(raw_path), "learning-results__" + features_params_string)
    # print("Saving learning results to {}...".format(results_path))
    results_df.to_csv(results_path)

    # print([['mean_test_score', 'mean_train_score']])
    # results_to_print = results_df.sort_values(['real_test_score'], ascending=False)
    # .loc[:, ['real_test_score', 'mean_test_score', 'mean_train_score']].head(1).values.tolist()

    print(results_df.sort_values(['real_test_score'], ascending=False)
          .loc[:, ['real_test_score', 'mean_test_score', 'mean_train_score', 'train_types', 'val_types', 'test_type']]
          .head(1))

    # print('Test: %0.2f, Validation: %0.2f, Train: %0.2f' %
    # (results_to_print[0][0], results_to_print[0][1], results_to_print[0][2]), features_params_string)
    return results_df


def extract_features(
        raw_path,
        au_selection_method,
        au_top_n,
        feature_selection_method,
        features_top_n,
        pca_method,
        pca_dimension,
        learning_method,
        features_params_string
):

    features_path = path.join(path.dirname(raw_path), "features__" + features_params_string)
    au_cor_path = path.join(path.dirname(raw_path), "au-correlation__" + features_params_string)
    features_cor_path = path.join(path.dirname(raw_path), "features-correlation__" + features_params_string)

    print("Reading {}...".format(raw_path))
    raw_df = pd.read_csv(raw_path)

    print("Choosing Top AU with method", au_selection_method)
    top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)

    print("Extracting features with method:", feature_selection_method)
    top_features = utils.get_top_features(top_AU, feature_selection_method, features_top_n, learning_method, raw_path)

    print("Saving top AU and Features to {} , {} ...".format(au_cor_path, features_cor_path))
    utils.get_corr_(top_AU, au_top_n, au_selection_method).to_csv(au_cor_path)
    utils.get_corr_(top_features, features_top_n, feature_selection_method).to_csv(features_cor_path)

    print("Saving all features to {}...".format(features_path))
    top_features.to_csv(features_path)

    if pca_method is not None:
        pca_path = path.join(path.dirname(raw_path), "pca_" + path.basename(raw_path))

        top_features = utils.dimension_reduction(pca_dimension, pca_method, top_features)
        top_features.to_csv(pca_path)

    return top_features


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('-i', '--input', dest='raw_path', required=True)

    # AU selection method
    parser.add_argument('-a', '--au_selection_method', dest='au_selection_method', type=str, default='top',
                        choices=list(AU_SELECTION_METHODS.values()))

    # Number of top correlated AUs to use if au_choice='top' selected
    parser.add_argument('-an', '--au_top_n', dest='au_top_n', type=int, default=24)

    # Feature selection method
    parser.add_argument('-f', '--feature_selection_method', dest='feature_selection_method', type=str, default=None,
                        choices=list(FEATURE_SELECTION_METHODS.values()))

    # Number of top correlated features to use
    parser.add_argument('-fn', '--features_top_n', dest='features_top_n', type=int, default=24)

    # Dimensionality reduction method.
    # 'global' runs PCA on all data
    # 'grouped' runs on each group of features separately (predefined in extract_features())
    parser.add_argument('-pm', '--pca_method', dest='pca_method', type=str, default=None,
                        choices=[None] + list(PCA_METHODS.values()))

    # Dimension after reduction.
    # If reduction is global, this is the final dimension
    # If reduction is by groups, this is a dimension for each group
    parser.add_argument('-p', '--pca_dim', dest='pca_dim', type=int, default=None)

    # Folding/learning method
    parser.add_argument('-lm', '--learning_method', dest='learning_method', type=str,
                        default=SPLIT_METHODS['4_vs_1_all_session_types'],
                        choices=list(SPLIT_METHODS.values()))

    parser.add_argument('-m', '--metric', dest='metric', type=str, default=None)

    parser.add_argument('-ts', '--take_sessions', dest='take_sessions', type=int, default=None,
                        choices=list(range(2, 21, 2)))

    parser.add_argument('-MR', '--mega_runner', dest='mega_runner', action='store_true')

    args = parser.parse_args()

    if args.mega_runner:


        def mega_run(raw_path, au_selection_method, feature_selection_method, pca_method, learning_method, metric, take_sessions):
            raw_df = pd.read_csv(args.raw_path)

            for au_top_n in range(15, 25):
                top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)

                for features_top_n in range(24, 80):
                    top_features = utils.get_top_features(top_AU, feature_selection_method, features_top_n, learning_method, raw_path)

                    if pca_method == PCA_METHODS['global']:
                        pca_options = range(5, 11)
                    else:
                        pca_options = range(1,6)

                    for pca_dim in pca_options:
                        try:

                            ext_features = utils.dimension_reduction(pca_dim, pca_method, top_features)

                            features_params_string = '[{}][au-method={}][au-top-n={}][fe-method={}][fe-top-n={}][pca-dim={}][pca-method={}][learning-method={}]'.format(
                                path.basename(raw_path),
                                au_selection_method,
                                au_top_n,
                                feature_selection_method,
                                features_top_n,
                                pca_dim,
                                pca_method,
                                learning_method
                            )

                            cv_method_all_learners(
                                raw_path,
                                ext_features,
                                learning_method,
                                metric,
                                features_params_string,
                                take_sessions
                            )
                        except Exception as e:
                            print('----\t[Error]\t' + str(e))


        if args.au_selection_method is None:
            print('No AU selection method provided')
            pass

        else:  # use given au selection method
            au_selection_method = args.au_selection_method

            if args.feature_selection_method is None:
                print('No feature selection method provided')
                pass

            else:  # use given feature selection method
                feature_selection_method = args.feature_selection_method

                if args.pca_method is None:
                    print('No PCA method provided')
                    pass

                else:  # use given pca method
                    pca_method = args.pca_method

                    if args.learning_method is None:
                        print('No learning method provided')
                        pass

                    else:  # use given learning method
                        learning_method = args.learning_method

                        mega_run(args.raw_path, au_selection_method, feature_selection_method, pca_method, learning_method, args.metric, args.take_sessions)

        # TODO run extract_features and cv_all_learners in lots of "for"s
        # take only YES or only NO from all sessions
        # last raw answer is really really long
        # take only first pair of sessions
        # don't learn other classifiers, only SVC
        # try to learn only on second answer (note the dataset may be imbalanced, weight it)

        # ress = []

        # m = args.learning_method

        # # print("Reading {}...".format(raw_path))
        # raw_df = pd.read_csv(args.raw_path)

        # for au_selection_method in AU_SELECTION_METHODS.values():

        #     for au_top_n in range(10, 25):
        #         # print("Choosing Top AU with method", au_selection_method)
        #         top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, m)

        #         for feature_selection_method in FEATURE_SELECTION_METHODS.values():

        #             for features_top_n in range(20, 81):
        #                 # print("Extracting features with method:", feature_selection_method)
        #                 top_features = utils.get_top_features(top_AU, feature_selection_method, features_top_n, m, args.raw_path)

        #                 for pca_method in list(PCA_METHODS.values()) + [None]:
        #                     pca_opts = range(5, 20)

        #                     for pca_dim in pca_opts:
        #                         try:

        #                             features_params_string = 'input_{}_au-method_{}_au-top-n_{}_f-method_{}_f-top-n_{}_pca-dim_{}_pca-method_{}_learning-method_{}'.format(
        #                               path.basename(args.raw_path),
        #                               au_selection_method,
        #                               au_top_n,
        #                               feature_selection_method,
        #                               features_top_n,
        #                               pca_dim,
        #                               pca_method,
        #                               m
        #                             )

        #                             features_path = path.join(path.dirname(args.raw_path), "features__" + features_params_string)
        #                             au_cor_path = path.join(path.dirname(args.raw_path), "au-correlation__" + features_params_string)
        #                             features_cor_path = path.join(path.dirname(args.raw_path), "features-correlation__" + features_params_string)

        #                             utils.get_cor(top_AU, au_top_n, au_selection_method).to_csv(au_cor_path)
        #                             utils.get_cor(top_features, features_top_n, feature_selection_method).to_csv(features_cor_path)
        #                             top_features.to_csv(features_path)
        #                             pca_features = top_features

        #                             # return_path = features_path

        #                             if pca_method is not None:
        #                                 pca_path = path.join(path.dirname(args.raw_path), "pca_" + features_params_string)

        #                                 pca_features = utils.dimension_reduction(pca_dim, pca_method, pca_path, top_features)
                                        
        #                                 # print("Saving PCA features to {}...".format(pca_path), end="")
        #                                 pca_features.to_csv(pca_path)

        #                                 # return_path = pca_path

        #                             ress.append(
        #                                 cv_method_all_learners(
        #                                     args.raw_path,
        #                                     pca_features,
        #                                     m,
        #                                     args.metric,
        #                                     features_params_string,
        #                                     args.take_sessions
        #                                 )
        #                             )

        #                         except Exception as e:
        #                             print('---- ERROR ----\r\n' + str(e) + '\r\n')

    else:

        if args.pca_method and args.pca_dim is None:
            parser.error("PCA method (-pm/--pca_method) requires dimension (-p/--pca_dim)")
            exit()

        features_params_string = 'input_{}_au-method_{}_au-top-n_{}_f-method_{}_f-top-n_{}_pca-dim_{}_pca-method_{}_learning-method_{}.csv'.format(
            path.basename(args.raw_path),
            args.au_selection_method,
            args.au_top_n,
            args.feature_selection_method,
            args.features_top_n,
            args.pca_dim,
            args.pca_method,
            args.learning_method
        )

        ext_features = extract_features(
            args.raw_path,
            args.au_selection_method,
            args.au_top_n,
            args.feature_selection_method,
            args.features_top_n,
            args.pca_method,
            args.pca_dim,
            args.learning_method,
            features_params_string
        )
        #features_path = 'pca_new_jonathan.csv'

        cv_method_all_learners(
            args.raw_path,
            ext_features,
            args.learning_method,
            args.metric,
            features_params_string,
            args.take_sessions
        )

#"""
#import os
#os.chdir('/cs/engproj/3deception/eran/3deception')
#import argparse
#import pandas as pd
#import os.path as path
#from features import utils
#raw_path = 'new_jonathan.csv'
#features_path = path.join(path.dirname(raw_path), "features_" + path.basename(raw_path))
#au_selection_method = 'top'
#au_top_n = 24
#feature_selection_method = 'groups'
#features_top_n = 80
#pca_method = 'groups'
#pca_dim = 6
#learning_method ='4v1_T'
#print("Reading {}...".format(raw_path))
#raw_df = pd.read_csv(raw_path)
#print("Choosing Top AU with method", au_selection_method)
#top_AU = utils.get_top_au(raw_df, au_selection_method, au_top_n, learning_method)
#print("Extracting features with method:", feature_selection_method)
#top_features = utils.get_top_features(top_AU, feature_selection_method, features_top_n, learning_method, raw_path)
#print("Saving all features to {}...".format(features_path), end="")
#top_features.to_csv(features_path)
#"""
