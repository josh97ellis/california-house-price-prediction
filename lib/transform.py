"""
Custom tranformer and machine learning pipeline to process and
fit a scikit-learn estimator
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.neighbors import KNeighborsRegressor

rooms_idx, bedrooms_idx, population_idx, households_idx = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    """
    Attribute Combinations for better model performance
    """

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rooms_per_household = X[:, population_idx] / X[:, households_idx]
        population_per_household = X[:, population_idx] / X[:, households_idx]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_idx] / X[:, rooms_idx]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def transformation_pipeline(X_train, y_train, model=KNeighborsRegressor()):
    """
    Defines the transformations required for numerical and categorical
    attributes. Any scikit-learn model can be fit to this pipeline
    given the model parameter

    Params:
    ----------------------------------------------------------------
    X_train: A set of data containing the attributes for predictions
    y_train: A set of labels which the model is learning to predict
    model: The specific estimator class to be fitted
    -> Example: LinearRegression(), KNeighborsRegressor(), ...
    """
    # Define Numerical Columns
    numeric_attributes = list(X_train.drop('ocean_proximity',  axis=1))

    # Define Categorical Columns
    categorical_attributes = ['ocean_proximity']

    # Define Numerical Pipeline
    numerical_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('attributes_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])

    # Define Categorical Pipeline
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse=False))
    ])

    # Combine Categorical and Numerical Pipeline
    preprocessor = ColumnTransformer([
        ('cat', categorical_pipeline, categorical_attributes),
        ('num', numerical_pipeline, numeric_attributes)
    ])

    # Fit a pipeline with transformers and an estimator to the training data
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    return model_pipeline.fit(X_train, y_train)
