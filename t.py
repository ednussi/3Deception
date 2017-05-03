import pandas as pd
df = pd.read_csv('eran_22-4-17_17-00-00.csv')
from enum import IntEnum
class RecordFlags(IntEnum):
    RECORD_FLAG_EMPTY = 0
    RECORD_FLAG_PAUSE = 1
    RECORD_FLAG_QUESTION = 2
    RECORD_FLAG_ANSWER_TRUE = 3
    RECORD_FLAG_ANSWER_FALSE = 4
    RECORD_FLAG_CONTROL_SHAPE_TRUE = 5
    RECORD_FLAG_CONTROL_SHAPE_FALSE = 6
    RECORD_FLAG_START_SESSION = 7
    RECORD_FLAG_END_SESSION = 8
    RECORD_FLAG_CHANGE = 9
    RECORD_FLAG_EXAMPLE_QUESTION = 10
    RECORD_FLAG_EXAMPLE_ANSWER_TRUE = 11
    RECORD_FLAG_EXAMPLE_ANSWER_FALSE = 12
    RECORD_FLAG_CATCH_ITEM = 13
DROP_COLUMNS = ['question', 'record_flag', 'record_index']
ANSWER_FLAGS = [RecordFlags.RECORD_FLAG_ANSWER_TRUE, RecordFlags.RECORD_FLAG_ANSWER_FALSE]
SKIP_COLUMNS = len(DROP_COLUMNS)
answers_df = df[df.record_flag.astype(int).isin(ANSWER_FLAGS)]
qdfs = [t[1] for t in answers_df[answers_df.record_index != 2].groupby(DROP_COLUMNS)]
from tsfresh import extract_features
fdfs = []
for d in qdfs:
    fdfs.append(d.drop(['record_flag', 'record_index'], axis=1))
ddfs, y = [], []
for i, x in enumerate(fdfs):
    ddfs.append(extract_features(x, column_id="question", column_sort="timestamp"))
    y.append(qdfs[i].record_flag.iloc[0])
