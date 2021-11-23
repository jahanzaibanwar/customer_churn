"""
 Customer Churn Model tests
 author: Jahanzaib
 date: Nov 18, 2021
 """

import logging
import os
import churn_library as cls
from churn_library import CATEGORICAL_COLUMNS
#from churn_library import import_data
import pytest

logging.basicConfig(
    filename='logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')



def test_import():
    """
    test data import - this example is completed for you to assist with the
    other test functions
    """
    try:
        df = cls.import_data("data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and "
            "columns")
        raise err
    return df



def test_eda():
    df = test_import()
    """
    test perform eda function
    """
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    cls.perform_eda(df)
    path = 'images/eda'
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info("Testing folder eda: EDA")
    except AssertionError as err:
        logging.warning("Testing perform_eda: It does not appear that the you "
                        "are correctly saving images to the eda folder.")
        raise err
    return df


def test_encoder_helper():
    """
    test encoder helper
    """
    df = test_eda()
    #cat_columns = ['Gender', 'Education_Level', 'Marital_Status',
     #              'Income_Category', 'Card_Category'
     #              ]
    # df = cls.encoder_helper(df,cat_columns,'Churn')
    # print(df)
    try:
        for cat_cols in CATEGORICAL_COLUMNS:
            df = cls.encoder_helper(df, cat_cols, 'Churn')
            assert cat_cols in df.columns
        logging.info("Testing encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: The dataframe appears to be missing the "
            "transformed categorical columns")
        return err

    return df

#
def test_perform_feature_engineering():
    """
    test perform_feature_engineering
    """
    df = test_encoder_helper()
    X_train_data, X_test_data, y_train_data, y_test_data = \
        cls.perform_feature_engineering(df)
    try:
        assert X_train_data.shape[0] == y_train_data.shape[0]
        assert X_train_data.shape[0] > 1
        assert X_test_data.shape[0] > 1
        assert len(y_train_data) > 1
        assert len(y_test_data) > 1

        logging.info('Testing perform_feature_engineering: SUCCESS ')
    except AssertionError as err:
        logging.info('The train test split is not valid')
        raise err
    return X_train_data, X_test_data, y_train_data, y_test_data

#
def test_train_models():
    """
    test train_models
    """
    X_train_data, X_test_data, y_train_data, y_test_data \
        =test_perform_feature_engineering()
    cls.train_models(X_train_data, X_test_data, y_train_data, y_test_data)
    path = 'images/results'
    try:
        dir_val = os.listdir(path)
        assert len(dir_val) > 0
        logging.info('Train models image results found : SUCCESS ')
    except AssertionError as err:
        logging.info('Train models image results not found : SUCCESS ')
        raise err
    model_path = 'models'
    try:
        model_dir = os.listdir(model_path)
        assert len(model_dir) > 0
        logging.info('Train models stored results found : SUCCESS ')
    except AssertionError as err:
        logging.info('Train models stored results not found : SUCCESS ')
        raise err


#DATAFRAME = test_import()
#test_eda(DATAFRAME)
#test_encoder_helper()
#X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering( )
#test_train_models()


if __name__ == "__main__":
    DATAFRAME = test_import()

    test_eda()
    test_encoder_helper()

    X_TRAIN, X_TEST, Y_TRAIN, Y_TEST = test_perform_feature_engineering( )
    test_train_models()
