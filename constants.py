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
    '4_vs_1_ast_single_answer': '4v1_SINGLE',
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