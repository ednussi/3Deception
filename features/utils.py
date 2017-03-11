import pandas as pd


def split_df_to_questions(df: pd.DataFrame) -> list(pd.DataFrame):
    """
    Split raw data frame by questions
    """
    return [*map(lambda x: x[1], df.drop(['timestamp'], axis=1).groupby(['question']))]


def scale(val):
    """
    Scale the given value from the scale of val to the scale of bot-top.
    """
    bot = 0
    top = 1

    if max(val)-min(val) == 0:
        return val
    return ((val - min(val)) / (max(val)-min(val))) * (top-bot) + bot

