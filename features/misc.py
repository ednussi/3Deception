import pandas as pd
from . import utils


def misc(question_dfs):
    # Calculated by taking the maximum of the amount of peaks in the
    # signals of both lip corners, where peak is defined as a local minimum which is higher by at
    # least 0.75 as compared to its surrounding points.
    num_smiles = []

    # The amount of blinks was calculated in a
    # similar manner, with a threshold of 0.2.
    num_blinks = []

    # Mean volume level
    mean_audio_rms = []

    # Maximal volume level
    max_audio_rms = []

    # Time delay till first continuous window with high volume
    # The audio signal is first smoothed to remove narrow peaks
    audio_rms_delay = []

    for ans in question_dfs:
        th = 0.75  # smile maximum threshold
        # Smiles
        # TODO Maybe separate to left-right-center smile
        num_smiles.append(max(utils.count_peaks(ans.loc[:, 'MouthSmile_L'].tolist(), delta=th),
                              utils.count_peaks(ans.loc[:, 'MouthSmile_R'].tolist(), delta=th)))

        # Blinks
        # TODO Maybe separate to left-right blinks
        num_blinks.append(max(utils.count_peaks(ans.loc[:, 'EyeBlink_L'].tolist()),
                              utils.count_peaks(ans.loc[:, 'EyeBlink_R'].tolist())))

        # Mean volume
        mean_audio_rms.append(ans.loc['audio_rms'].mean())

        # Max volume
        max_audio_rms.append(ans.loc['audio_rms'].max())



    df = pd.DataFrame({'smiles': num_smiles, 'blinks': num_blinks})
    return df
