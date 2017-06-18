SUBJECTS = [
    'alon',
    'arielt',
    'asaph',
    'inbar',
    'liav',
    'lilach',
    'maayan',
    'omri',
    'yuval'
]
from os import listdir, getcwd, path
import pandas as pd
# os.chdir(r'\users\gregory\projects\3deception')
method = '4v1_T.max'
for subj in SUBJECTS:
    subj_dir = path.join(getcwd(), 'questionnaire', 'data', subj)
    subj_raw_path = path.join(subj_dir, list(filter(lambda x: x.startswith('fs_'), listdir(subj_dir)))[0])
    subj_results_path = path.join(subj_dir, list(filter(lambda x: x.endswith('.{}.csv'.format(method)), listdir(subj_dir)))[0])

    if (path.isfile(subj_results_path)):
        df = pd.read_csv(subj_results_path)
        for t in range(1, 6):
            print(subj, t)
            tdf = df[df.test_type == '[{}]'.format(t)]
            ttdf = tdf.sort_values(['mean_test_score'], ascending=False).head(20)
            ttdf['diff'] = abs(ttdf.mean_test_score - ttdf.mean_train_score)
            ttdf['subj'] = subj
            tttdf = ttdf.sort_values(['diff']).head(1)
            tttdf.to_csv(subj + '_type' + str(t) + '.csv')
            del tdf
            del ttdf
            del tttdf
        del df

cols = ['mean_fit_time', 'mean_score_time', 'mean_test_score', 'mean_train_score', 'param_C', 'param_kernel', 'params', 'rank_test_score', 'split0_test_score', 'split0_train_score', 'split1_test_score', 'split1_train_score', 'split2_test_score', 'split2_train_score', 'split3_test_score', 'split3_train_score', 'std_fit_time', 'std_score_time', 'std_test_score', 'std_train_score', 'best_mean_val_score', 'train_types', 'val_type', 'test_type', 'best_estimator_train_score', 'best_estimator_test_score', 'au_method', 'au_top', 'fe_method', 'fe_top', 'learning_method', 'pca_dim', 'pca_method', 'lm2', 'subj']
dd = []
for subj in SUBJECTS:
    for t in range(1, 6):
        dd.append(pd.read_csv(subj + '_type' + str(t) + '.csv').loc[:, cols])
fd = pd.concat(dd)
ddd = pd.DataFrame(columns=fd.subj.unique(), index=fd.test_type.unique())
for subj in ddd.columns:
    for t in ddd.index:
        ffd = fd[fd.subj == subj]
        print(ffd.shape)
        v = ffd[ffd.test_type == t]
        print(v.shape)
        ddd.loc[t, subj] = v['best_estimator_test_score'].values.tolist()[0]
ddd.to_csv('all_res_t_new_max.csv')
