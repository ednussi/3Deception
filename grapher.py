import argparse
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.gridspec as gridspec
from mpl_toolkits.mplot3d import Axes3D
from features import utils
from constants import PCA_METHODS, SPLIT_METHODS, AU_SELECTION_METHODS, FEATURE_SELECTION_METHODS, META_COLUMNS, TARGET_COLUMN, RecordFlags


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    print('Reading arguments..')
    # results csv
    parser.add_argument('-r', '--results', dest='results_path')

    args = parser.parse_args()

    print('Loading csv..')
    res_df = pd.read_csv(args.results_path, index_col=0)

    print('Plotting graphs..')
    fig = plt.figure(1)
    gs = gridspec.GridSpec(3, 3)



    for fig_num, test_q_type in enumerate(res_df['test_type'].unique()):
        ######################### MULTI DIM GRAPH #########################

        fig = plt.figure(2)
        print('Plotting graph {}..'.format(fig_num + 1))
        #Filter out
        df = res_df[res_df['test_type'] == '[{}]'.format(fig_num+1)]

        fig = plt.figure(fig_num)
        fig.subplots_adjust(hspace=0.8)
        fig.subplots_adjust(wspace=0.8)

        for i, au_top_num in enumerate(df['au_top'].unique()):

            df_au = df[df['au_top'] == au_top_num]
            row = i // 3
            col = i % 3
            ax = plt.subplot2grid((3, 3), (row, col))

            ax.set_xlabel('Top Features')
            ax.set_ylabel('PCA Dim')

            #
            C_res = df_au['param_C']
            size = ((C_res - C_res.min()) / (C_res.max() - C_res.min())).pow(3)

            #
            test_score = df_au['best_estimator_test_score']
            normalize_res_df = (test_score - 0.5) / 0.35
            color = plt.cm.RdYlGn(normalize_res_df)



            # Scatter the dots
            print('Scattering The {} dots..'.format(len(color)))
            ax.scatter(df_au['fe_top'], df_au['pca_dim'], s=size, color=color)
            plt.title('AU top {}'.format(au_top_num))

        fig_name = args.results_path + '_MULTI_GRAPH_{}.png'.format(test_q_type)
        print('Saving fig into:{}..'.format(fig_name))
        plt.xticks(rotation=90)
        plt.savefig(fig_name)

        print('Grapher Finished!')

    ######################### MULTI DIM GRAPH #########################



ax1 = plt.subplot2grid(shape=(2,6), loc=(0,0), colspan=2)
ax2 = plt.subplot2grid((2,6), (0,2), colspan=2)
ax3 = plt.subplot2grid((2,6), (0,4), colspan=2)
ax4 = plt.subplot2grid((2,6), (1,1), colspan=2)
ax5 = plt.subplot2grid((2,6), (1,3), colspan=2)