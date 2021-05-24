from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd


def features(df):
    """ Function to preprocess dataframe.
      Args:
             df (numpy.array): raw dataframe
      Returns:
             df (numpy.array): raw dataframe

         """
    # list all categorical features that we want to encode using OneHotEncoder
    categorical_features = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                            'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
    encoder = OneHotEncoder()
    encoded = pd.DataFrame(encoder.fit_transform(df[categorical_features]).toarray(),
                           columns=encoder.get_feature_names(categorical_features))
    df = df.drop(columns=categorical_features).join(encoded)
    # list all numerical features that we want to re-scale
    numeric_features = ['count_floors_pre_eq', 'area_percentage', 'height_percentage', 'count_families']
    df[numeric_features] = MinMaxScaler().fit_transform(df[numeric_features])

    return df


def feature_engineering(df):
    """ Function to generate additional features.
         Args:
             df (numpy.array): raw dataframe
         Returns:
             df (numpy.array): raw dataframe

         """

    # Creation of dummy features degree of destruction of the different districts
    df.loc[(df['geo_level_1_id'] == 14) |
           (df['geo_level_1_id'] == 15) | (df['geo_level_1_id'] == 2) |
           (df['geo_level_1_id'] == 23) | (df['geo_level_1_id'] == 28) |
           (df['geo_level_1_id'] == 19) | (
                   df['geo_level_1_id'] == 29), 'district_class_1'] = 1
    df.loc[(df['geo_level_1_id'] != 14) &
           (df['geo_level_1_id'] != 15) & (df['geo_level_1_id'] != 2) &
           (df['geo_level_1_id'] != 23) & (df['geo_level_1_id'] != 28) &
           (df['geo_level_1_id'] != 19) & (
                   df['geo_level_1_id'] != 29), 'district_class_1'] = 0

    df.loc[(df['geo_level_1_id'] == 30) | (df['geo_level_1_id'] == 24) |
           (df['geo_level_1_id'] == 0) | (df['geo_level_1_id'] == 1) |
           (df['geo_level_1_id'] == 16) | (df['geo_level_1_id'] == 12) |
           (df['geo_level_1_id'] == 18) |
           (df['geo_level_1_id'] == 5), 'district_class_2'] = 1
    df.loc[(df['geo_level_1_id'] != 0) & (df['geo_level_1_id'] != 30) &
           (df['geo_level_1_id'] != 24) & (df['geo_level_1_id'] != 1) &
           (df['geo_level_1_id'] != 12) &
           (df['geo_level_1_id'] != 16) & (df['geo_level_1_id'] != 18) &
           (df['geo_level_1_id'] != 5), 'district_class_2'] = 0

    df.loc[(df['geo_level_1_id'] == 4) | (df['geo_level_1_id'] == 20) |
           (df['geo_level_1_id'] == 26), 'district_class_3'] = 1
    df.loc[(df['geo_level_1_id'] != 4) & (df['geo_level_1_id'] != 20) &
           (df['geo_level_1_id'] != 26), 'district_class_3'] = 0

    df.loc[(df['geo_level_1_id'] == 6) | (df['geo_level_1_id'] == 7) |
           (df['geo_level_1_id'] == 8) | (df['geo_level_1_id'] == 10) |
           (df['geo_level_1_id'] == 21) | (df['geo_level_1_id'] == 17)
    , 'district_class_4'] = 1
    df.loc[(df['geo_level_1_id'] != 6) & (df['geo_level_1_id'] != 7) &
           (df['geo_level_1_id'] != 8) & (df['geo_level_1_id'] != 10) &
           (df['geo_level_1_id'] != 21) & (
                   df['geo_level_1_id'] != 17), 'district_class_4'] = 0

    # Creation of dummy features on the age of the buildings
    df.loc[(df['age'] <= 40), 'age_u_40'] = 1
    df.loc[(df['age'] > 40), 'age_u_40'] = 0
    df.loc[(df['age'] > 40) & (df['age'] <= 100), 'age_40_100'] = 1
    df.loc[(df['age'] <= 40) | (df['age'] > 100), 'age_40_100'] = 0
    df.loc[(df['age'] >= 100), 'age_ue_100'] = 1
    df.loc[(df['age'] < 100), 'age_ue_100'] = 0

    # rescale age and geo_features after generating dummy features
    scale_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'age']
    df[scale_features] = MinMaxScaler().fit_transform(df[scale_features])

    return df


def split_data(df):
    """ Function to split dataframe into train and test data.
     Args:
         df (numpy.array): raw dataframe
     Returns:
         train_data (numpy.array): preprocessed dataframe
         train_target (numpy.array): labels of target data
         test_data (numpy.array): preprocessed test dataframe

     """
    df = df.set_index('building_id')
    # split data back into train,label and test
    train_data = df.loc[np.invert(df.damage_grade.isna())]
    train_target = train_data['damage_grade']
    train_data = train_data.drop(columns='damage_grade')
    test_data = df.loc[df.damage_grade.isna()].drop(columns='damage_grade')

    return train_data, train_target, test_data
