import pandas as pd


def import_data(train_set='../data/external/train_values.csv',
                train_labes='../data/external/train_labels.csv',
                test_set='../data/external/test_labels.csv'
                ):
    """ Function to import the data and concatenate it into a dataframe.
    Args:
        train_set (csv): dataframe containing features
        train_labels (csv): dataframe containing labels
        test_set (csv): dataframe containing features
    Returns:
        dataframe with train and test features, only train_set has labels
    """
    train_values = pd.read_csv('../data/external/train_values.csv', index_col='building_id')
    train_target = pd.read_csv('../data/external/train_labels.csv', index_col='building_id')
    test_values = pd.read_csv('../data/external/test_values.csv', index_col='building_id')

    data = train_values.join(train_target)

    return pd.concat([data, test_values])
