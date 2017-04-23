import pandas as pd
from . import utils


PEAK_THRESHOLD = 0.75
ROLLING_WINDOW = 30


def misc(question_dfs):
    # Calculated by taking the maximum of the amount of peaks in the
    # signals of both lip corners, where peak is defined as a local minimum which is higher by at
    # least 0.75 as compared to its surrounding points.
    num_smiles = []

    # The amount of blinks was calculated in a
    # similar manner, with a threshold of 0.2.
    num_blinks = []

    # Mean volume level
    audio_rms_mean = []

    # Maximal volume level
    audio_rms_max = []

    # Time delay till first continuous window with high volume
    # The audio signal is first smoothed to remove narrow peaks
    audio_rms_delay = []

    for ans in question_dfs:
        # Smiles
        # TODO Maybe separate to left-right-center smile
        num_smiles.append(max(utils.count_peaks(ans.loc[:, 'MouthSmile_L'].tolist(), delta=PEAK_THRESHOLD),
                              utils.count_peaks(ans.loc[:, 'MouthSmile_R'].tolist(), delta=PEAK_THRESHOLD)))

        # Blinks
        # TODO Maybe separate to left-right blinks
        num_blinks.append(max(utils.count_peaks(ans.loc[:, 'EyeBlink_L'].tolist()),
                              utils.count_peaks(ans.loc[:, 'EyeBlink_R'].tolist())))

        audio_rms_mean.append(ans.loc[:, 'audio_rms'].mean())
        audio_rms_max.append(ans.loc[:, 'audio_rms'].max())

        rolling = ans.loc[:, 'audio_rms'].rolling(window=ROLLING_WINDOW, center=False, min_periods=1).mean()
        threshold_rolling = (rolling > rolling.mean() + rolling.std()).values.tolist()

        current_length, j = 1, 0
        sequences = []
        prev = threshold_rolling[0]

        for i, x in enumerate(threshold_rolling[1:]):
            if x != prev:
                sequences.append((prev, j, current_length))
                current_length = 1
                j = i + 1
            else:
                current_length += 1
            prev = x

        sequences.append((prev, j, current_length))

        # delay is the length of silent segment before last noisy one
        # intuition: the last noisy segment is
        idx = len(sequences) - 2 - [*map(lambda x: x[0], sequences)][::-1].index(1)

        # in case there is only one noisy segment (the subject says NO in the time of recording), ignore
        if idx >= 0:
            audio_rms_delay.append(sequences[idx][2])
        else:
            audio_rms_delay.append(0)

    df = pd.DataFrame({
        'smiles': num_smiles, 
        'blinks': num_blinks,
        'audio_rms_mean': audio_rms_mean,
        'audio_rms_max': audio_rms_max,
        'response_delay': audio_rms_delay
    })

    return df
