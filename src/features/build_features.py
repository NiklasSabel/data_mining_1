from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd


def features(df):
    """
    Function to preprocess dataframe.
    :param df: raw dataframe, type: pd.DataFrame()
    :return: preprocessed dataframe with OneHotEncoder, MinMaxScaler
    """

    # list all categorical features that we want to encode using OneHotEncoder
    categorical_features = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                            'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
    encoder = OneHotEncoder()
    encoded = pd.DataFrame(encoder.fit_transform(df[categorical_features]).toarray(),
                           columns=encoder.get_feature_names(categorical_features))
    df = df.drop(columns=categorical_features).join(encoded)

    # list all numerical features that we want to re-scale
    numeric_features = ['count_floors_pre_eq', 'count_families']
    df[numeric_features] = MinMaxScaler().fit_transform(df[numeric_features])

    return df


def split_data(df):
    """
    Function to split dataframe into train and test data.
    :param df: Preprocessed dataframe with added means of categorical features and dropped correlated features
    :return: tuple of preprocessed train data, labels of target data (train target) and preprocessed test data splitted
    """

    df = df.set_index('building_id')

    # split data back into train, label and test
    train_data = df.loc[np.invert(df.damage_grade.isna())]  # data where labels for damage grade are empty
    train_target = train_data['damage_grade']
    train_data = train_data.drop(columns='damage_grade')
    test_data = df.loc[df.damage_grade.isna()].drop(columns='damage_grade')

    return train_data, train_target, test_data


def feature_engineering(df):
    """
    Function to generate additional features.
    :param df: merged dataframe of preprocessed train, test and target data
    :return: dataframe with new features
    """

    # Creation of dummy features degree of destruction of the different districts

    district_class_1 = [2, 14, 15, 19, 23, 28, 29]
    district_class_2 = [0, 1, 5, 12, 16, 18, 24, 30]
    district_class_3 = [4, 20, 26]
    district_class_4 = [6, 7, 8, 10, 17, 21]

    for i in range(1, 31):
        if i in district_class_1:
            df.loc[df['geo_level_1_id'] == i, 'district_class_1'] = 1
        elif i in district_class_2:
            df.loc[df['geo_level_1_id'] == i, 'district_class_2'] = 1
        elif i in district_class_3:
            df.loc[df['geo_level_1_id'] == i, 'district_class_3'] = 1
        elif i in district_class_4:
            df.loc[df['geo_level_1_id'] == i, 'district_class_4'] = 1

    df[['district_class_1', 'district_class_2', 'district_class_3', 'district_class_4']] = \
        df[['district_class_1', 'district_class_2', 'district_class_3', 'district_class_4']].fillna(0)

    # Creation of dummy features on the age of the buildings
    df.loc[(df['age'] <= 40), 'age_u_40'] = 1
    df.loc[(df['age'] > 40), 'age_u_40'] = 0
    df.loc[(df['age'] > 40) & (df['age'] <= 100), 'age_40_100'] = 1
    df.loc[(df['age'] <= 40) | (df['age'] > 100), 'age_40_100'] = 0
    df.loc[(df['age'] >= 100), 'age_ue_100'] = 1
    df.loc[(df['age'] < 100), 'age_ue_100'] = 0

    # Creation of dummy features indicating geo_levels with high and low mud_mortar_stone
    low_mortar_percentage = []
    high_mortar_percentage = []

    for i in range(0, 31):

        geolvl_count = df[df['geo_level_1_id'] == i]['geo_level_1_id'].value_counts().item()
        mortar = df[df['geo_level_1_id'] == i]['has_superstructure_mud_mortar_stone'].sum()

        mortar_percentage = (mortar / geolvl_count)

        if mortar_percentage < 0.6:
            low_mortar_percentage.append(i)
        else:
            high_mortar_percentage.append(i)

    # dummy creation
    column_low = []
    for val in df['geo_level_1_id']:
        if val in low_mortar_percentage:
            column_low.append(1)
        else:
            column_low.append(0)

    column_high = []
    for val in df['geo_level_1_id']:
        if val in high_mortar_percentage:
            column_high.append(1)
        else:
            column_high.append(0)

    df['low_mortar_percentage'] = column_low

    df['high_mortar_percentage'] = column_high

    # Creation of dummy features for identity regions with high share of foundation type r -> assumption high damage
    low_percentage_r = []
    high_percentage_r = []

    for i in range(0, 31):
        geolvl_count = df[df['geo_level_1_id'] == i]['geo_level_1_id'].value_counts().item()
        type_r = df[df['geo_level_1_id'] == i]['foundation_type_r'].sum()

        percentage = ((type_r) / geolvl_count)

        if percentage < 0.6:
            low_percentage_r.append(i)
        else:
            high_percentage_r.append(i)

    # dummy creation
    column_low = []
    for val in df['geo_level_1_id']:
        if val in low_percentage_r:
            column_low.append(1)
        else:
            column_low.append(0)

    column_high = []
    for val in df['geo_level_1_id']:
        if val in high_percentage_r:
            column_high.append(1)
        else:
            column_high.append(0)

    df['low_percentage_r'] = column_low
    df['high_percentage_r'] = column_high

    # creation of feature fragile with bad building material and stable with good building material
    fragile = ['other_floor_type_q', 'foundation_type_r',
               'ground_floor_type_f']
    stable = ['ground_floor_type_v', 'other_floor_type_j', 'foundation_type_w',
              'has_superstructure_cement_mortar_brick', 'roof_type_x']

    for feature in fragile:
        df.loc[df[feature] == 1, 'fragile'] = 1

    for feature in stable:
        df.loc[df[feature] == 1, 'stable'] = 1

    df[['fragile', 'stable']] = df[['fragile', 'stable']].fillna(0)

    # Adding ft_importance 1 --> features that are rather stable so that the damage grade is mostly 1
    ft_importance_1_pos = ['has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
                           'has_secondary_use', 'has_secondary_use_hotel', 'foundation_type_u', 'foundation_type_w',
                           'roof_type_x', 'other_floor_type_s']
    ft_high_importance_1_pos = ['has_superstructure_cement_mortar_brick', 'ground_floor_type_w', 'other_floor_type_j']

    # feature importance 1 positive
    df.loc[(df['has_superstructure_rc_non_engineered'] == 1) | (df['has_superstructure_rc_engineered'] == 1) |
           (df['has_secondary_use'] == 1) | (df['has_secondary_use_hotel'] == 1) |
           (df['foundation_type_u'] == 1) | (df['foundation_type_w'] == 1) | (df['roof_type_x'] == 1) |
           (df['other_floor_type_s'] == 1), 'ft_imp_1_pos'] = 1

    # high feature importance 1 positive
    df.loc[(df['has_superstructure_cement_mortar_brick'] == 1) | (df['ground_floor_type_v'] == 1) |
           (df['other_floor_type_j'] == 1), 'ft_high_imp_1_pos'] = 1

    # fill nans of positive feature importances with 0 & nans of negative feature importance with 1
    df[['ft_imp_1_pos', 'ft_high_imp_1_pos']] = \
        df[['ft_imp_1_pos', 'ft_high_imp_1_pos']].fillna(0)

    # add feature with the population density in every geo_level_id
    df['dens_1'] = df.groupby('geo_level_1_id')['geo_level_1_id'].transform('size')
    df['dens_2'] = df.groupby('geo_level_2_id')['geo_level_2_id'].transform('size')
    df['dens_3'] = df.groupby('geo_level_3_id')['geo_level_3_id'].transform('size')

    # rescale age and geo_features after generating dummy features, except geo_level_1_id
    scale_features = ['geo_level_2_id', 'geo_level_3_id', 'dens_1', 'dens_2', 'dens_3']

    df[scale_features] = MinMaxScaler().fit_transform(df[scale_features])

    return df


def get_unnecessary_ft(df):
    """
    Function to get all unnecessary features which lay under a threshold of 0.01 relative occurrence per damage grade each
    :param df: general dataframe
    :return: list with all unnecessary features
    """

    dmg = df.groupby('damage_grade').agg({'damage_grade': 'count'})
    df = df.groupby('damage_grade').sum()
    df = df.join(dmg)
    df = df.iloc[:, :-1].div(df.damage_grade, axis=0)
    df = df.transpose()

    unnecessary_ft = []

    for i in range(df.shape[0]):  # iterate through rows
        liste = []
        for j in range(df.shape[1]):  # iterate through columns
            value = df.iloc[i, j]
            if value > 0.01:
                liste.append(value)
        if len(liste) == 0:
            unnecessary_ft.append(df.index[i])

    return unnecessary_ft


def drop_unnecessary_ft(df):
    """
    Function to drop all unnecessary features which lay under a threshold of 0.01 relative occurrence per damage grade each
    :param df: general dataframe
    :return: dataframe without unnecessary features
    """

    df = df.drop(columns=get_unnecessary_ft(df), axis=1)

    return df


def drop_correlated_features(df, thr=0.8):
    """
    Function to detect all correlated features
    :param df: general dataframe and threshold for the correlation param
    :return: dataframe without correlated features
    """
    correlated_features = set()
    correlation_matrix = df.corr()

    for i in range(len(correlation_matrix.columns)):
        for j in range(i):
            if abs(correlation_matrix.iloc[i, j]) > thr:
                print(
                    f"The following features are correlated: {correlation_matrix.columns[i]} and {correlation_matrix.columns[j]}. Correlation = {round(abs(correlation_matrix.iloc[i, j]), 2)}")
                colname = correlation_matrix.columns[j]
                correlated_features.add(colname)
    print(f"Drop the following features: {correlated_features}")
    # drop correlated features
    df = df.drop(columns=correlated_features)

    return df
