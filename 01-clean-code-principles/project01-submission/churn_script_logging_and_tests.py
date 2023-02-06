"""
Module that perfoms tests & logging for churn_library.py file 

Author: Christoph Schmidl
Date: January 29, 2023
"""

import glob
import logging
import os
import sys
from pathlib import Path

import joblib
import pandas as pd
import pytest

import churn_library as cls
from constants import cat_columns

# Some weird stuff with logging and pytest
# Pytest has its own logging system and it's better
# to define everything in the pytest.ini file
# https://github.com/pytest-dev/pytest/issues/9989
logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')

###################################
#           Fixtures
###################################

# scope="module" -> only invoked once per test module (the default is to
# invoke once per test function).


@pytest.fixture(scope="module")
def raw_data():
    """
    raw data fixture - returns the raw data as a Pandas dataframe
    """
    try:
        raw_df = cls.import_data("data/bank_data.csv")
        logging.info("Created raw data as dataframe fixture: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Raw data as dataframe fixture: The file wasn't found.")
        raise err

    return raw_df


@pytest.fixture(scope="module")
def encoded_data():
    """
    raw data fixture - returns the raw data as a Pandas dataframe
    """
    try:
        raw_df = cls.import_data("data/bank_data.csv")
        encoded_df = cls.encoder_helper(raw_df, cat_columns, 'Churn')
        logging.info("Created encoded data as dataframe fixture: SUCCESS")
    except FileNotFoundError as err:
        logging.error(
            "Encoded data as dataframe fixture: The file wasn't found or encoding failed.")
        raise err

    return encoded_df


@pytest.fixture(scope="module")
def import_data():
    """
    Fixture for method "import_data"
    """
    return cls.import_data


@pytest.fixture(scope="module")
def perform_eda():
    """
    Fixture for method "perform_eda"
    """
    return cls.perform_eda


@pytest.fixture(scope="module")
def encoder_helper():
    """
    Fixture for method "encoder_helper"
    """
    return cls.encoder_helper


@pytest.fixture(scope="module")
def perform_feature_engineering():
    """
    Fixture for method "perform_feature_engineering"
    """
    return cls.perform_feature_engineering


@pytest.fixture(scope="module")
def train_models():
    """
    Fixture for method "train_models"
    """
    return cls.train_models


###################################
#           Unit tests
###################################

def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
    except FileNotFoundError as err:
        # log error
        logging.error("Testing import_data: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
        logging.info("Testing import_data: SUCCESS")  # log success
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(perform_eda, raw_data):
    '''
    test perform eda function
    '''
    # 1. Execute EDA
    perform_eda(raw_data)

    # 2. Check if images exist
    eda_path = Path("./images/eda")

    for image_name in [
        "churn_distribution.png",
        "corr_heatmap.png",
        "customer_age_distribution.png",
        "marital_status_distribution.png",
            "total_transaction_distribution.png"]:
        image_path = eda_path.joinpath(image_name)
        try:
            assert image_path.is_file()
        except AssertionError as err:
            logging.error(
                "Testing perform_eda: The file wasn't found or is not of type file.")
            raise err
    logging.info("Testing perform_eda: SUCCESS")  # log success


def test_encoder_helper(encoder_helper, raw_data):
    '''
    test encoder helper
    '''
    encoded_data = encoder_helper(raw_data, cat_columns, 'Churn')

    assert isinstance(encoded_data, pd.DataFrame)
    try:
        assert encoded_data.shape[0] > 0
        assert encoded_data.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe does not contain rows and columns.")
        raise err

    try:
        for element in cat_columns:
            assert element in encoded_data.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error("Testing encoder_helper: Some categorial columns are missing.")
        return err


def test_perform_feature_engineering(
        perform_feature_engineering,
        encoded_data):
    '''
    test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        encoded_data, 'Churn')

    try:
        # check shape and length
        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Shape or length mismatch.")
        raise err


def test_train_models(train_models, encoded_data):
    '''
    test train_models
    '''
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        encoded_data, 'Churn')
    train_models(X_train, X_test, y_train, y_test)

    # 1. Check if images exist
    results_path = Path("./images/results")

    for image_name in [
        "feature_importances.png",
        "lr_results.png",
        "rf_results.png",
            "ROC_curves.png"]:
        image_path = results_path.joinpath(image_name)
        try:
            assert image_path.is_file()
        except AssertionError as err:
            logging.error(
                "Testing train_models: The image file wasn't found or is not of type file.")
            raise err

    # 2. Check if models exist
    models_path = Path("./models")

    for model_name in ["logistic_model.pkl", "rfc_model.pkl"]:
        model_path = models_path.joinpath(model_name)
        try:
            assert model_path.is_file()
        except AssertionError as err:
            logging.error(
                "Testing train_models: The model file wasn't found or is not of type file.")
            raise err
    logging.info("Testing train_models: SUCCESS")  # log success


def clear_dirs():
    logging.info("Clearing directories...")
    for directory in ["logs", "images/eda", "images/results", "models"]:
        files = glob.glob(f"{directory}/*")
        for file in files:
            os.remove(file)


if __name__ == "__main__":
    clear_dirs()
