import pandas as pd
from . import utils


PEAK_THRESHOLD = 0.75
ROLLING_WINDOW = 30


def misc(question_dfs):
    # Calculated by taking the maximum of the amount of peaks in the
    # signals of both lip corners, where peak is defined as a local minimum which is higher by at
    # least 0.75 as compared to its surrounding points.
    num_smiles = []
    num_smiles_right = []
    num_smiles_left = []

    # The amount of blinks was calculated in a
    # similar manner, with a threshold of 0.2.
    num_blinks = []
    num_blinks_right = []
    num_blinks_left = []

    # Time delay till first continuous window with high volume
    # The audio signal is first smoothed to remove narrow peaks
    audio_rms_delay = []

    for ans in question_dfs:
        # ================== Smiles ==================
        if ('MouthSmile_L' in ans.columns):
            num_smiles_left.append(utils.count_peaks(ans.loc[:, 'MouthSmile_L'].tolist(), delta=PEAK_THRESHOLD))
            if ('MouthSmile_R' in ans.columns):
                num_smiles_right.append(utils.count_peaks(ans.loc[:, 'MouthSmile_R'].tolist(), delta=PEAK_THRESHOLD))
                num_smiles.append(max(utils.count_peaks(ans.loc[:, 'MouthSmile_L'].tolist(), delta=PEAK_THRESHOLD),
                                  utils.count_peaks(ans.loc[:, 'MouthSmile_R'].tolist(), delta=PEAK_THRESHOLD)))
            else:
                num_smiles_right.append(0)
                num_smiles.append(0)
        elif ('MouthSmile_R' in ans.columns):
            num_smiles_right.append(utils.count_peaks(ans.loc[:, 'MouthSmile_R'].tolist(), delta=PEAK_THRESHOLD))
            num_smiles_left.append(0)
            num_smiles.append(0)
        else:
            num_smiles_right.append(0)
            num_smiles_left.append(0)
            num_smiles.append(0)


        # ================== Blinks ==================
        if ('EyeBlink_L' in ans.columns):
            num_blinks_left.append(utils.count_peaks(ans.loc[:, 'EyeBlink_L'].tolist()))
            if ('EyeBlink_R' in ans.columns):
                num_blinks_right.append(utils.count_peaks(ans.loc[:, 'EyeBlink_R'].tolist()))
                num_blinks.append(max(utils.count_peaks(ans.loc[:, 'EyeBlink_L'].tolist()),
                                      utils.count_peaks(ans.loc[:, 'EyeBlink_R'].tolist())))
            else:
                num_blinks_right.append(0)
                num_blinks.append(0)
        elif ('EyeBlink_R' in ans.columns):
            num_blinks_right.append(utils.count_peaks(ans.loc[:, 'EyeBlink_R'].tolist()))
            num_blinks_left.append(0)
            num_blinks.append(0)
        else:
            num_blinks_right.append(0)
            num_blinks_left.append(0)
            num_blinks.append(0)



                # ================== Response delay ==================
        # audio = ans.loc[:, 'audio_rms']
        # threshold_audio = (audio > 1).values.tolist()
        #
        # current_length, j = 1, 0
        # sequences = []
        # prev = threshold_audio[0]
        #
        # for i, x in enumerate(threshold_audio[1:]):
        #     if x != prev:
        #         sequences.append((prev, j, current_length))
        #         current_length = 1
        #         j = i + 1
        #     else:
        #         current_length += 1
        #     prev = x
        #
        # sequences.append((prev, j, current_length))
        #
        # # delay is the length of silent segment before last noisy one
        # # intuition: the last noisy segment is
        # idx = len(sequences) - 2 - [*map(lambda x: x[0], sequences)][::-1].index(True)
        #
        # # in case there is only one noisy segment (the subject says NO in the time of recording), ignore
        # if idx >= 0:
        #     audio_rms_delay.append(sequences[idx][2])
        # else:
        #     audio_rms_delay.append(0)
        #
        # # ================== Response duration ==================
        # idx = len(sequences) - 1 - [*map(lambda x: x[0], sequences)][::-1].index(True)
        # response_dur = sequences[idx][2]

    df = pd.DataFrame({
        'smiles': num_smiles,
        'smiles_right': num_smiles_right,
        'smiles_left': num_blinks_left,
        'blinks': num_blinks,
        'blinks_right': num_blinks_right,
        'blinks_left': num_blinks_left,
        # 'response_delay': audio_rms_delay,
        # 'response_duration': response_dur
    })

    return df
