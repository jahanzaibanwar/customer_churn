"""
Customer Churn Analysis and Modelling

author: Jahanzaib
date: Nov 18, 2021
"""

# import libraries
import numpy as np
import pandas as pd
# import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    data = pd.read_csv(pth)
    data['Churn'] = data['Attrition_Flag'].apply(lambda val: 0 if val == 'Existing Customer' else 1)
    return data


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    print(df.isnull().sum(), "\n")
    print("Shape of the dataframe is {}".format(df.shape), "\n")
    # print(df.describe())

    column_names = ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct']

    for col in column_names[:-2]:
        plt.figure(figsize=(20, 10))
        df[col].hist()
        plt.savefig('images/{}_histogram'.format(col))
        plt.close()

    plt.figure(figsize=(20, 10))
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig('images/Marital_Status_Value Count')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.histplot(df['Total_Trans_Ct'], kde=True)
    plt.savefig('images/Total_Trans_Ct Distribution')
    plt.close()

    plt.figure(figsize=(20, 10))
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig('images/Dataframe Correlation Heatmap')
    plt.close()


dataframe = import_data('data/bank_data.csv')
perform_eda(dataframe)


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """
    value_list = []
    column_groups = df.groupby(category_lst).mean()[response]

    for val in df[category_lst]:
        value_list.append(column_groups.loc[val])

    df[category_lst + '_Churn'] = value_list

    return df


target_column = 'Churn'
column_list = ['Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category']
encoded_column = pd.DataFrame(np.NAN)
for col in column_list:
    encoded_column = encoder_helper(dataframe, col, target_column)

print(encoded_column['Card_Category_Churn'])

print(encoded_column.columns)


def perform_feature_engineering(df, response):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """


def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    """
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    pass


def feature_importance_plot(model, X_data, output_pth):
    """
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure

    output:
             None
    """
    pass


def train_models(X_train, X_test, y_train, y_test):
    """
    train, store model results: images + scores, and store models
    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    """
    pass
