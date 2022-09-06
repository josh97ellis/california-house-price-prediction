"""
Provides functionality to automate the retrieval,
storage, and loading of source data from online
"""
import os
import tarfile
import urllib.request
import pandas as pd


def fetch_housing_data():
    """
    Gets california housing prices data from github URL
    and loads into into a local directory
    """
    download_root = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
    housing_url = download_root + "datasets/housing/housing.tgz"
    housing_path = os.path.join("datasets")

    # Make file directory to store the data
    os.makedirs(
        housing_path,
        exist_ok=True)

    tgz_path = os.path.join(
        housing_path,
        "housing.tgz")

    # Obtain data from github
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)

    # Load data into local file directory
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def read_housing_data():
    """
    Reads california housing prices data from local directory
    """
    # File to file
    housing_path = os.path.join("datasets")
    csv_path = os.path.join(housing_path, "housing.csv")

    # Load data into pandas dataframe
    return pd.read_csv(csv_path)


if __name__ == '__main__':
    print('Executing from __main__')
    print('-> Fetching Data')
    fetch_housing_data()
    print('-> Reading Data')
    read_housing_data()
    print('Done')
