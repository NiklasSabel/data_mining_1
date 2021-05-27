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
    ###
    #df[numeric_features] = StandardScaler().fit_transform(df[numeric_features])

    return df


def split_data(df):
    """
    Function to split dataframe into train and test data.
    :param df: Preprocessed dataframe with added means of categorical features and dropped correlated features
    :return: tuple of preprocessed train data, labels of target data (train target) and preprocessed test data splitted
    """

    df = df.set_index('building_id')

    # split data back into train, label and test
    train_data = df.loc[np.invert(df.damage_grade.isna())] # data where labels for damage grade are empty
    train_target = train_data['damage_grade']
    train_data = train_data.drop(columns='damage_grade')
    test_data = df.loc[df.damage_grade.isna()].drop(columns='damage_grade')

    return train_data, train_target, test_data


def feature_engineering(df,geo_district_nu):
    """
    Function to generate additional features.
    :param df: merged dataframe of preprocessed train, test and target data
    :return: dataframe with new features
    """

    # Creation of dummy features degree of destruction of the different districts
    if geo_district_nu == 3:

        district_class_1 = [6, 10, 13, 20, 26]
        district_class_2 = [3, 4, 6, 7, 8, 10, 11, 13, 20, 21, 22, 25, 26, 27]
        district_class_3 = [6, 7, 8, 10, 11, 17, 21, 27]

        for i in range(1, 31):
            if i in district_class_1:
                df.loc[df['geo_level_1_id'] == i, 'district_class_1'] = 1
            elif i in district_class_2:
                df.loc[df['geo_level_1_id'] == i, 'district_class_2'] = 1
            elif i in district_class_3:
                df.loc[df['geo_level_1_id'] == i, 'district_class_3'] = 1

        df[['district_class_1', 'district_class_2', 'district_class_3']] = \
            df[['district_class_1', 'district_class_2', 'district_class_3']].fillna(0)

    elif geo_district_nu == 4:

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

    # Identity regions with high share of foundation type r -->assumption high damage

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

    # Define ``stable`` and ``fragile`` constructions

    # Create dummies

    # create fragile construction dummy
    df.loc[(df['foundation_type_r'] == 1) & (df['other_floor_type_q'] == 1) & (
            df['ground_floor_type_f'] == 1), 'fragile'] = 1
    df.loc[(df['foundation_type_r'] == 1) & (df['other_floor_type_q'] == 1) & (
            df['ground_floor_type_f'] == 0), 'fragile'] = 1
    df.loc[(df['foundation_type_r'] == 1) & (df['other_floor_type_q'] == 0) & (
            df['ground_floor_type_f'] == 1), 'fragile'] = 1
    df.loc[(df['foundation_type_r'] == 0) & (df['other_floor_type_q'] == 1) & (
            df['ground_floor_type_f'] == 1), 'fragile'] = 1
    df.loc[(df['foundation_type_r'] == 1) & (df['other_floor_type_q'] == 0) & (
            df['ground_floor_type_f'] == 0), 'fragile'] = 0
    df.loc[(df['foundation_type_r'] == 0) & (df['other_floor_type_q'] == 1) & (
            df['ground_floor_type_f'] == 0), 'fragile'] = 0
    df.loc[(df['foundation_type_r'] == 0) & (df['other_floor_type_q'] == 0) & (
            df['ground_floor_type_f'] == 1), 'fragile'] = 0
    df.loc[(df['foundation_type_r'] == 0) & (df['other_floor_type_q'] == 0) & (
            df['ground_floor_type_f'] == 0), 'fragile'] = 0

    # create stable construction dummy
    df.loc[(df['foundation_type_i'] == 1) & (df['ground_floor_type_v'] == 1) & (
            df['roof_type_x'] == 1), 'stable'] = 1
    df.loc[(df['foundation_type_i'] == 1) & (df['ground_floor_type_v'] == 1) & (
            df['roof_type_x'] == 0), 'stable'] = 1
    df.loc[(df['foundation_type_i'] == 1) & (df['ground_floor_type_v'] == 0) & (
            df['roof_type_x'] == 1), 'stable'] = 1
    df.loc[(df['foundation_type_i'] == 0) & (df['ground_floor_type_v'] == 1) & (
            df['roof_type_x'] == 1), 'stable'] = 1
    df.loc[(df['foundation_type_i'] == 1) & (df['ground_floor_type_v'] == 0) & (
            df['roof_type_x'] == 0), 'stable'] = 0
    df.loc[(df['foundation_type_i'] == 0) & (df['ground_floor_type_v'] == 1) & (
            df['roof_type_x'] == 0), 'stable'] = 0
    df.loc[(df['foundation_type_i'] == 0) & (df['ground_floor_type_v'] == 0) & (
            df['roof_type_x'] == 1), 'stable'] = 0
    df.loc[(df['foundation_type_i'] == 0) & (df['ground_floor_type_v'] == 0) & (
            df['roof_type_x'] == 0), 'stable'] = 0

    # rescale age and geo_features after generating dummy features
    # scale_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'age']
    # df[scale_features] = MinMaxScaler().fit_transform(df[scale_features])

    # Adding ft_importance 1
    ft_importance_1_neg = ['has_superstructure_mud_mortar_stone', 'foundation_type_r', 'roof_type_n',
                           'ground_floor_type_f',
                           'other_floor_type_q']
    ft_importance_1_pos = ['has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
                           'has_secondary_use',
                           'has_secondary_use_hotel', 'foundation_type_u', 'foundation_type_w', 'roof_type_x',
                           'other_floor_type_s']
    ft_high_importance_1_pos = ['has_superstructure_cement_mortar_brick', 'ground_floor_type_w', 'other_floor_type_j']

    df.loc[(df['has_superstructure_rc_non_engineered'] == 1) | (df['has_superstructure_rc_engineered'] == 1) |
           (df['has_secondary_use'] == 1) | (df['has_secondary_use_hotel'] == 1) |
           (df['foundation_type_u'] == 1) | (df['foundation_type_w'] == 1) | (df['roof_type_x'] == 1) |
           (df['other_floor_type_s'] == 1), 'ft_imp_1_pos'] = 1
    df.loc[(df['has_superstructure_rc_non_engineered'] != 1) & (df['has_superstructure_rc_engineered'] != 1) &
           (df['has_secondary_use'] != 1) & (df['has_secondary_use_hotel'] != 1) &
           (df['foundation_type_u'] != 1) & (df['foundation_type_w'] != 1)
           & (df['roof_type_x'] != 1) & (df['other_floor_type_s'] != 1), 'ft_imp_1_pos'] = 0

    df.loc[(df['has_superstructure_cement_mortar_brick'] == 1) | (df['ground_floor_type_v'] == 1) |
           (df['other_floor_type_j'] == 1), 'ft_high_imp_1_pos'] = 1
    df.loc[(df['has_superstructure_cement_mortar_brick'] != 1) & (df['ground_floor_type_v'] != 1) &
           (df['other_floor_type_j'] != 1), 'ft_high_imp_1_pos'] = 0

    return df


def get_unnecessary_ft (df):
    dmg = df.groupby('damage_grade').agg({'damage_grade': 'count'})
    df = df.groupby('damage_grade').sum()
    df = df.join(dmg)
    df = df.iloc[:, :-1].div(df.damage_grade, axis=0)
    df = df.transpose()

    unnecessary_ft = []
    for i in range(df.shape[0]):  # rows
        liste = []
        for j in range(df.shape[1]):  # columns
            value = df.iloc[i, j]
            if value > 0.01:
                liste.append(value)
        if len(liste) == 0:
            unnecessary_ft.append(df.index[i])

    return unnecessary_ft

def drop_unnecessary_ft(df):
    df = df.drop(columns=get_unnecessary_ft(df), axis=1)
    return df
