import pandas as pd
from utils import count_peaks



def miscl(question_dfs):
    '''
    # of Smiles
    Calculated by taking the maximum of the amount of peaks in the
    signals of both lip corners, where peak is defined as a local minimum which is higher by at
    least 0.75 as compared to its surrounding points.

    # of Blinks
    The amount of blinks was calculated in a
    similar manner, with a threshold of 0.2.

    '''

    num_smiles =[]
    num_blinks = []
    ans_miscl = []
    for ans in question_dfs:
        th = 0.75 #smile maximum threshold
        # Smiles
        mouth_smiles = ans.loc[:, 'MouthSmile_L':'MouthSmile_R']
        # TODO Maybe seperate to left-right-center smile
        num_smiles.append(max(count_peaks(ans.loc[:, 'MouthSmile_L'].tolist(), delta=th),
            count_peaks(ans.loc[:, 'MouthSmile_R'].tolist(), delta=th)))

        # Blinks
        blinks = ans.loc[:, 'EyeBlink_L':'EyeBlink_R']
        # TODO Maybe seperate to left-right blinks
        num_blinks.append(max(count_peaks(ans.loc[:, 'EyeBlink_L'].tolist()),
            count_peaks(ans.loc[:, 'EyeBlink_R'].tolist())))

    df = pd.DataFrame({'smiles':num_smiles, 'blinks':num_blinks})
    #ans_miscl.append(pd.concat(pd.DataFrame([[num_smiles, num_blinks]], columns=['smiles', 'blinks'])))
    return df



