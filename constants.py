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

META_COLUMNS = [
    'session',
    'session_type',
    'question', 
    'question_type', 
    'record_flag',
    'answer_index', 
    'timestamp'
]

GROUPBY_COLUMNS = [
    'question',
    'answer_index'
]

SESSION_COLUMN = 'session'
SESSION_TYPE_COLUMN = 'session_type'
QUESTION_COLUMN = 'question'
QUESTION_TYPE_COLUMN = 'question_type'
ANSWER_INDEX_COLUMN = 'answer_index'
TARGET_COLUMN = 'record_flag'

ANSWER_FLAGS = [
    RecordFlags.RECORD_FLAG_ANSWER_TRUE,
    RecordFlags.RECORD_FLAG_ANSWER_FALSE
]

SESSION_TYPES = {
    'say_truth': 1,
    'say_lies': 2,
}

QUESTION_TYPES = {
    'first_name': 1,
    'surname': 2,
    'mother_name': 3,
    'birth_country': 4,
    'birth_month': 5,
}

SPLIT_METHODS = {
    '4_vs_1_all_session_types': '4v1_A',
    '4_vs_1_truth_session_types': '4v1_T',
    '4_vs_1_lies_session_types': '4v1_F',
    '4_vs_1_single_answer': '4v1_SINGLE',
    'session_prediction': 'SP'
}


PCA_METHODS = {
    'global': 'global',
    'groups': 'groups'
}


AU_SELECTION_METHODS = {
    'daniel': 'daniel',
    'mouth': 'mouth',
    'eyes': 'eyes',
    'brows': 'brows',
    'eyes_area': 'eyes_area',
    'smile': 'smile',
    'blinks': 'blinks',
    'top': 'top'
}


FEATURE_SELECTION_METHODS = {
    'all': 'all',
    'groups': 'groups'
}


__all__ = [
    'RecordFlags',
    'META_COLUMNS',
    'GROUPBY_COLUMNS',
    'SESSION_COLUMN',
    'SESSION_TYPE_COLUMN',
    'QUESTION_TYPE_COLUMN',
    'TARGET_COLUMN',
    'ANSWER_FLAGS',
    'SESSION_TYPES',
    'QUESTION_TYPES',
    'SPLIT_METHODS'
]