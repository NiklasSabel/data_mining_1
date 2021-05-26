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
    # get the mean damage grade of different geo_levels
    df = df.merge(df.groupby(['geo_level_1_id']).mean()['damage_grade'], on='geo_level_1_id', how='left').rename(
        columns={"damage_grade_x": 'damage_grade', "damage_grade_y": "mean_dmg_geo_1"}).merge(df.groupby(['geo_level_2_id']).mean()['damage_grade'], on='geo_level_2_id',
                                                                                              how='left').rename(
        columns={"damage_grade_x": 'damage_grade', "damage_grade_y": "mean_dmg_geo_2"})
    df = df.merge(df.groupby(['foundation_type']).mean()['damage_grade'], on='foundation_type', how='left').rename(
        columns={"damage_grade_x": 'damage_grade', "damage_grade_y": "mean_dmg_fnd_t"})
    df = df.merge(df.groupby(['roof_type']).mean()['damage_grade'], on='roof_type', how='left').rename(
        columns={"damage_grade_x": 'damage_grade', "damage_grade_y": "mean_dmg_roof_t"})
    df = df.merge(df.groupby(['ground_floor_type']).mean()['damage_grade'], on='ground_floor_type', how='left').rename(
        columns={"damage_grade_x": 'damage_grade', "damage_grade_y": "mean_dmg_ground_t"})
    df = df.merge(df.groupby(['other_floor_type']).mean()['damage_grade'], on='other_floor_type', how='left').rename(
        columns={"damage_grade_x": 'damage_grade', "damage_grade_y": "mean_dmg_other_floor_t"})
    df = df.merge(df.groupby(['legal_ownership_status']).mean()['damage_grade'], on='legal_ownership_status',
                  how='left').rename(
        columns={"damage_grade_x": 'damage_grade', "damage_grade_y": "mean_dmg_own_status_t"})
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


def feature_engineering_geo_3(df):
    """ Function to generate additional features.
         Args:
             df (numpy.array): raw dataframe
         Returns:
             df (numpy.array): raw dataframe

         """

    # Creation of dummy features degree of destruction of the different districts
    df.loc[(df['geo_level_1_id'] == 6) |
                     (df['geo_level_1_id'] == 10) | (df['geo_level_1_id'] == 13) |
                     (df['geo_level_1_id'] == 20) | (df['geo_level_1_id'] == 26)
    , 'district_class_1'] = 1
    df.loc[(df['geo_level_1_id'] != 6) &
                     (df['geo_level_1_id'] != 10) & (df['geo_level_1_id'] != 13) &
                     (df['geo_level_1_id'] != 20) & (df['geo_level_1_id'] != 26)
    , 'district_class_1'] = 0

    df.loc[(df['geo_level_1_id'] == 3) | (df['geo_level_1_id'] == 4) |
                     (df['geo_level_1_id'] == 6) | (df['geo_level_1_id'] == 7) |
                     (df['geo_level_1_id'] == 8) | (df['geo_level_1_id'] == 10) |
                     (df['geo_level_1_id'] == 11) | (df['geo_level_1_id'] == 13) |
                     (df['geo_level_1_id'] == 20) | (df['geo_level_1_id'] == 21) |
                     (df['geo_level_1_id'] == 22) | (df['geo_level_1_id'] == 25) |
                     (df['geo_level_1_id'] == 26) | (
                                 df['geo_level_1_id'] == 27), 'district_class_2'] = 1

    df.loc[(df['geo_level_1_id'] != 3) & (df['geo_level_1_id'] != 4) &
                     (df['geo_level_1_id'] != 6) & (df['geo_level_1_id'] != 7) &
                     (df['geo_level_1_id'] != 8) & (df['geo_level_1_id'] != 10) &
                     (df['geo_level_1_id'] != 11) & (df['geo_level_1_id'] != 13) &
                     (df['geo_level_1_id'] != 20) & (df['geo_level_1_id'] != 21) &
                     (df['geo_level_1_id'] != 22) & (df['geo_level_1_id'] != 25) &
                     (df['geo_level_1_id'] != 26) & (
                                 df['geo_level_1_id'] != 27), 'district_class_2'] = 0

    df.loc[(df['geo_level_1_id'] == 6) | (df['geo_level_1_id'] == 7) |
                     (df['geo_level_1_id'] == 8) | (df['geo_level_1_id'] == 10) |
                     (df['geo_level_1_id'] == 11) | (df['geo_level_1_id'] == 17) |
                     (df['geo_level_1_id'] == 21) | (
                                 df['geo_level_1_id'] == 27), 'district_class_3'] = 1
    df.loc[(df['geo_level_1_id'] != 6) & (df['geo_level_1_id'] != 7) &
                     (df['geo_level_1_id'] != 8) & (df['geo_level_1_id'] != 10) &
                     (df['geo_level_1_id'] != 11) & (df['geo_level_1_id'] != 17) &
                     (df['geo_level_1_id'] != 21) & (
                                 df['geo_level_1_id'] != 27), 'district_class_3'] = 0

    # Creation of dummy features on the age of the buildings
    df.loc[(df['age'] <= 40), 'age_u_40'] = 1
    df.loc[(df['age'] > 40), 'age_u_40'] = 0
    df.loc[(df['age'] > 40) & (df['age'] <= 100), 'age_40_100'] = 1
    df.loc[(df['age'] <= 40) | (df['age'] > 100), 'age_40_100'] = 0
    df.loc[(df['age'] >= 100), 'age_ue_100'] = 1
    df.loc[(df['age'] < 100), 'age_ue_100'] = 0

    ## Adding 2 dummies indicating geo_leovels with high and low mud_mortar_stone - share
    # low
    # high

    low_mortar_percentage = []
    high_mortar_percentage = []

    for i in range(1, 31):
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

    for i in range(1, 31):
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

def feature_engineering_geo_4(df):
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

    # Adding ft_importance 1
    ft_importance_1_neg = ['has_superstructure_mud_mortar_stone', 'foundation_type_r', 'roof_type_n',
                           'ground_floor_type_f',
                           'other_floor_type_q']
    ft_importance_1_pos = ['has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered',
                           'has_secondary_use',
                           'has_secondary_use_hotel', 'foundation_type_u', 'foundation_type_w', 'roof_type_x',
                           'other_floor_type_s']
    ft_high_importance_1_pos = ['has_superstructure_cement_mortar_brick', 'ground_floor_type_w', 'other_floor_type_j']

    for feature in ft_importance_1_pos:
        df.loc[(df[feature] == 1), 'ft_imp_1_pos'] = 1
        df.loc[(df[feature] != 1), 'ft_imp_1_pos'] = 0

    for feature in ft_high_importance_1_pos:
        df.loc[(df[feature] == 1), 'ft_high_imp_1_pos'] = 1
        df.loc[(df[feature] != 1), 'ft_high_imp_1_pos'] = 0

    ## Adding 2 dummies indicating geo_leovels with high and low mud_mortar_stone - share
    # low
    # high
    low_mortar_percentage = []
    high_mortar_percentage = []


    for i in range(1, 31):
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

    for i in range(1, 31):
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
    scale_features = ['geo_level_1_id', 'geo_level_2_id', 'geo_level_3_id', 'age']
    df[scale_features] = MinMaxScaler().fit_transform(df[scale_features])



def drop_unnecessary_ft (df):
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
