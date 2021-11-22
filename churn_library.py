# """
# Customer Churn Analysis and Modelling
#
# author: Jahanzaib
# date: Nov 18, 2021
# """
#


import logging

import joblib
# import shap
# import numpy as np
import matplotlib.pyplot as plt
# import libraries
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')

KEEP_COLUMNS = ['Customer_Age', 'Dependent_count', 'Months_on_book',
                'Total_Relationship_Count', 'Months_Inactive_12_mon',
                'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal',
                'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt',
                'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1',
                'Avg_Utilization_Ratio',
                'Gender_Churn', 'Education_Level_Churn',
                'Marital_Status_Churn',
                'Income_Category_Churn', 'Card_Category_Churn']

TARGET_COLUMN = 'Churn'
CATEGORICAL_COLUMNS = ['Gender', 'Education_Level', 'Marital_Status',
                       'Income_Category', 'Card_Category']


def import_data(pth):
    """
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    """
    data = pd.read_csv(pth)
    logging.info("Testing import_data: SUCCESS")
    data['Churn'] = data['Attrition_Flag'].apply(
        lambda val: 0 if val == 'Existing Customer' else 1)
    return data


def perform_eda(df):
    """
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    """
    # print(df.isnull().sum(), "\n")
    # print("Shape of the dataframe is {}".format(df.shape), "\n")
    column_list = ['Churn', 'Customer_Age', 'Marital_Status', 'Total_Trans_Ct']

    for col in column_list[:2]:
        print(col)
        print("*******************")
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


def encoder_helper(df, category_lst, response):
    """
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the
    notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be
            used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    """

    value_list = []
    column_groups = df.groupby(category_lst).mean()[response]
    for val in df[category_lst]:
        value_list.append(column_groups.loc[val])
    df[category_lst + '_Churn'] = value_list
    return df


def perform_feature_engineering(df):
    """
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could
              be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    """

    X = pd.DataFrame()
    y = df['Churn'].values

    X[KEEP_COLUMNS] = df[KEEP_COLUMNS]
    # print(len(X.columns))
    X = X.values
    X_train_data, X_test_data, y_train_data, y_test_data = \
        train_test_split(X, y, random_state=42)

    return X_train_data, X_test_data, y_train_data, y_test_data


def classification_report_image(y_train_data,
                                y_test_data,
                                y_train_preds_regression,
                                y_train_preds_random_forest,
                                y_test_preds_regression,
                                y_test_preds_random_forest):
    """
    produces classification report for training and testing results and
    stores report as image in images folder input: y_train: training
    response values y_test:  test response values y_train_preds_lr: training
    predictions from logistic regression y_train_preds_rf: training
    predictions from random forest y_test_preds_lr: test predictions from
    logistic regression y_test_preds_rf: test predictions from random forest

    output:
             None
    """
    plt.rc('figure', figsize=(9, 7))
    # plt.text(0.01, 0.05, str(model.summary()), {'fontsize': 12}) old approach
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test_data,
                                                   y_test_preds_random_forest))
             , {'fontsize': 10}, fontproperties='monospace')
    # approach improved by OP -># monospace!
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train_data,
                                                  y_train_preds_random_forest)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP ->
    # monospace!
    plt.axis('off')
    plt.savefig('images/Classification_Report Random Forest')
    plt.close()

    plt.rc('figure', figsize=(9, 7))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train_data,
                                                   y_train_preds_regression)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP ->
    # monospace!
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10},
             fontproperties='monospace')
    plt.text(0.01, 0.7,
             str(classification_report(y_test_data, y_test_preds_regression)),
             {'fontsize': 10},
             fontproperties='monospace')  # approach improved by OP ->
    # monospace!
    plt.axis('off')
    plt.savefig('images/Classification_Report Logistic Regression')
    plt.close()


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
    importances = model.best_estimator_.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 15))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=40)

    plt.savefig('{}/Feature Importance'.format(output_pth))
    plt.close()


def train_models(X_train_data, X_test_data, y_train_data, y_test_data):
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
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='liblinear')

    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5,
                          n_jobs=-1, refit=True)

    cv_rfc.fit(X_train_data, y_train_data)
    print()

    lrc.fit(X_train_data, y_train_data)

    # y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train_data)
    # y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test_data)
    #
    # y_train_preds_lr = lrc.predict(X_train_data)
    # y_test_preds_lr = lrc.predict(X_test_data)

    lrc_plot = metrics.plot_roc_curve(lrc, X_test_data, y_test_data)

    plt.savefig('images/Roc Curve lrc Test data')
    plt.close()

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    metrics.plot_roc_curve(cv_rfc.best_estimator_, X_test_data, y_test_data,
                           ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('images/Roc Curve lrc and rfc on Test data')
    plt.close()

    joblib.dump(cv_rfc, 'models/rfc_model.pkl')
    joblib.dump(lrc, 'models/logistic_model.pkl')


# return y_train_preds_rf, y_test_preds_rf, y_train_preds_lr,
# y_test_preds_lr


# y_train_prediction_rf, y_test_prediction_rf, y_train_prediction_lr,
# y_test_prediction_lr = train_models(
# X_train, X_test, y_train, y_test)
#
if __name__ == '__main__':
    # Reading the data
    dataframe = import_data('data/bank_data.csv')
    # Performing EDA on data
    perform_eda(dataframe)
    # Conversion of catgorical Columns
    # encoded_column = pd.DataFrame()
    for column_name in CATEGORICAL_COLUMNS:
        encoded_column = encoder_helper(dataframe, column_name,
                                        TARGET_COLUMN)
    print(encoded_column.columns)
    # Feature Enginering
    X_train, X_test, y_train, y_test = perform_feature_engineering \
        (encoded_column)
    print(X_train.shape)
    print(y_train.shape)

    # Training and saving the model
    train_models(X_train, X_test, y_train, y_test)
    # print("Done")

    # Feature Importance
    load_rfc_model = joblib.load('models/rfc_model.pkl')
    load_lr_model = joblib.load('models/logistic_model.pkl')
    output_path = 'images'

    feature_importance_plot(load_rfc_model, dataframe[KEEP_COLUMNS],
                            output_path)

    y_train_preds_rf = load_rfc_model.best_estimator_.predict(X_train)
    y_test_preds_rf = load_rfc_model.best_estimator_.predict(X_test)

    y_train_preds_lr = load_lr_model.predict(X_train)
    y_test_preds_lr = load_lr_model.predict(X_test)

    classification_report_image(y_train, y_test, y_train_preds_lr,
                                y_train_preds_rf, y_test_preds_lr,
                                y_test_preds_rf)
