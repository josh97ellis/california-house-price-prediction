"""
This module is resposible for training the model with
training data inputs. Once the model is saved, it is
saved as a .pkl file in the project directory.
"""
import data
import model_manager
import transform
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor

# Obtain a fresh set of data from online
data.fetch_housing_data()

# Read data as pandas dataframe
housing_df = data.read_housing_data()

# Split data into X (predictors) and y (labels)
X = housing_df.drop('median_house_value', axis=1)
y = housing_df['median_house_value'].copy()

# Partition the X and y datasets into test and training sets
X_train, X_test, y_train, y_test = (
    train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42)
)

# fit a KNeighborRegressor model
kn_reg = transform.transformation_pipeline(
    X_train,
    y_train,
    model=KNeighborsRegressor()
)

# Save the model to disk
model_manager.save_model(kn_reg)
