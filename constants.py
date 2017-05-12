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

GOOD_DANIEL_AU = [
    'EyeBlink_L',
    'EyeBlink_R',
    'EyeIn_L',
    'EyeIn_R',
    'BrowsU_C',
    'BrowsU_L',
    'BrowsU_R',
    'JawOpen',
    'MouthLeft',
    'MouthRight',
    'MouthFrown_L',
    'MouthFrown_R',
    'MouthSmile_L',
    'MouthSmile_R',
    'MouthDimple_L',
    'MouthDimple_R',
    'LipsStretch_L',
    'LipsStretch_R',
    'LipsUpperUp',
    'LipsFunnel',
    'ChinLowerRaise',
    'Sneer',
    'CheekSquint_L',
    'CheekSquint_R'
]

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
QUESTION_TYPE_COLUMN = 'question_type'
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
