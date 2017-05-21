import sys

path = sys.argv[-1]

import os
import pandas as pd

dfs = []
for f in os.listdir(path):
    if f.startswith('learning-results'):
        t = pd.read_csv(os.path.join(path,f))
        t['3dmodel'] = f
        dfs.append(t)

r = pd.concat(dfs)
rr = r.sort_values(['mean_test_score'], ascending=False)

rr.to_csv(os.path.join(path,'all_learning_results.csv'))
print(rr.loc[:, ['mean_test_score', '3dmodel']].head(5))
