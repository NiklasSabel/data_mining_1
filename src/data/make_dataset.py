import pandas as pd


def merge_data(train_values, train_target, test_values):
    """
    Function to import all data and concatenate it into one dataframe.
    :return: one dataframe only with train and test features together with train labels
    """

    data = train_values.join(train_target)

    return pd.concat([data, test_values])
