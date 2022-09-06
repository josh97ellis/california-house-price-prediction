"""
Module is executed for testing purposes only. Running
this module will randomly sample a single element
from the entire housing dataset. It will return
the result of the prediction as well as the actual
value from the data for comparison.
"""
from model_manager import load_model
from data import read_housing_data


def main():
    # Read housing price data from local directory
    housing_df = read_housing_data()

    # Split predictors(X) for labels(y)
    X = housing_df.drop('median_house_value', axis=1)
    y = housing_df['median_house_value'].copy()

    # Sample a random element from the housing data
    sample_idx = X.sample(n=1).index.item()
    X_input = X.filter(items=[sample_idx], axis=0)
    y_input = y.filter(items=[sample_idx], axis=0)

    loaded_model = load_model()

    y_predict = loaded_model.predict(X_input)[0]
    y_actual = y_input.item()

    print(f"Predicted Value: ${y_predict}")
    print(f"Actual House Value: ${y_actual}")
    print(f"Prediction is off by ${round((y_predict - y_actual), 2)}")


if __name__ == '__main__':
    main()
