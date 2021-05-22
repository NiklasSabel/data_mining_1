from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
import numpy as np
import pandas as pd


def features(df):
    """ Function to preprocess dataframe.
     Args:
         df (numpy.array): raw dataframe
     Returns:
         train_data (numpy.array): preprocessed dataframe
         train_target (numpy.array): labels of target data
         test_data (numpy.array): preprocessed test dataframe

     """
    # list all categorical features that we want to encode using OneHotEncoder
    categorical_features = ['land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type',
                            'other_floor_type', 'position', 'plan_configuration', 'legal_ownership_status']
    encoder = OneHotEncoder()
    encoded = pd.DataFrame(encoder.fit_transform(df[categorical_features]).toarray(),
                           columns=encoder.get_feature_names(categorical_features))
    df = df.drop(columns=categorical_features).join(encoded)
    # list all numerical features that we want to re-scale
    numeric_features = ['count_floors_pre_eq', 'age', 'area_percentage', 'height_percentage', 'count_families']
    scaler = MinMaxScaler()
    df[numeric_features] = scaler.fit_transform(df[numeric_features])
    df=df.set_index('building_id')
    # split data back into train,label and test
    train_data = df.loc[np.invert(df.damage_grade.isna())]
    train_target = train_data['damage_grade']
    train_data = train_data.drop(columns='damage_grade')
    test_data = df.loc[df.damage_grade.isna()].drop(columns='damage_grade')

    return train_data, train_target, test_data
