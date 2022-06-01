import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score,
                             precision_score,
                             recall_score,
                             classification_report, confusion_matrix,
                             precision_recall_curve)
import matplotlib.pyplot as plt

# Author: 210033137
# This function loads the dataset and returns the dataframe that we will use for model
def load_dataset(filename):
    # let try to load the data and set the column names
    names = ['ID', 'Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore',
             'Cscore', 'Impulsive', 'SS', ' Alcohol', ' Amphet', 'Amyl', 'Benzos', 'Caff', 'Cannabis', 'Choc', 'Coke',
             'Crack', 'Ecstasy', 'Heroin', 'Ketamine', 'Legalh', 'LSD', 'Meth', 'Mushrooms', 'Nicotine', 'Semer', 'VSA']
    df = pd.read_csv(filename, names=names)
    # lets extract the columns that we will use for models prediction
    df = df[['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Nscore', 'Escore', 'Oscore', 'Ascore',
             'Cscore', 'Impulsive', 'SS', 'Nicotine']]
    return df


# This function plots the histogram of all features
def analyse_data_histogram(df):
    df.hist(figsize=(15, 15))
    plt.show()


# In this function we are exploring the dataset, checking shape, unique values, and
# general information of dataset, null values, target distribution etc
def exploring_dataset(df):
    print(df.head())
    print("=" * 100)
    print("Shape of Dataset :", df.shape)
    print("=" * 100)
    print("Columns Names :\n", df.columns.tolist())
    print("=" * 100)
    # lets try to check the null values in the dataset
    print("Sum of Null values in Dataset :\n", df.isnull().sum())
    # lets try to check the general information of all columns in dataset
    print("General Information :\n", df.info())
    print("=" * 100)
    # lets try to check number of unique values in features
    c = 0
    for i in df.columns.tolist():
        c += 1
        print(f"Feature {c}: {i} ==> {df[i].nunique()} unique values")
    print("=" * 100)

    # lets try to check unique values in features
    c1 = 0
    for i in df.columns.tolist():
        c1 += 1
        print(f"Feature {c1}: {i} ==> {df[i].unique()} unique values")
    print("=" * 100)
    print("Target Feature Distribution :\n", df['Nicotine'].value_counts())
    print("=" * 100)


# In this function we are one hot encoding all the categorical features
def one_hotEncoding(df, columns):
    # Lets one hot encode the categorical features
    df_Age = pd.get_dummies(df[columns[0]], prefix=columns[0])
    df_Gender = pd.get_dummies(df[columns[1]], prefix=columns[1])
    df_Education = pd.get_dummies(df[columns[2]], prefix=columns[2])
    df_Country = pd.get_dummies(df[columns[3]], prefix=columns[3])
    df_Ethnicity = pd.get_dummies(df[columns[4]], prefix=columns[4])
    df_Impulsive = pd.get_dummies(df[columns[5]], prefix=columns[5])
    df_SS = pd.get_dummies(df[columns[6]], prefix=columns[6])
    df.drop(columns, axis=1, inplace=True)
    # This concatenates the one hot encoding features into new dataframe
    df_new = pd.concat([df, df_Age, df_Gender, df_Education,
                        df_Country, df_Ethnicity, df_Impulsive,
                        df_Impulsive, df_SS], axis=1)
    return df_new


# This function returns the input features and output feature for model
def extract_dependent_independent_features(df):
    X = df.drop('Nicotine', axis=1)  # input features except Nicotine
    y = df['Nicotine']  # Nicotine output feature
    y = y.map({'CL0': 0, 'CL1': 1, 'CL2': 2, 'CL3': 3, 'CL4': 4, 'CL5': 5, 'CL6': 6})
    return X, y


# This function we are splitting the dataset into training and testing
# By default we are using the 80% data for training and 20% for testing
def splitting_data(X, y, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)
    print("Training Shape: ", X_train.shape)
    print("Testing Shape: ", X_test.shape)
    return X_train, X_test, y_train, y_test


# This function rescale the data into range between 0 and 1. (normalize the data)
def scaling_data(X_train, X_test):
    std = StandardScaler()
    X_train_scaled = std.fit_transform(X_train)
    X_test_scaled = std.transform(X_test)
    return X_train_scaled, X_test_scaled


# In this function we are building a logistic regression with passing parameters
# We can check the model prediction on penalty and class weight and also we can
# Plot the precision recall curve on each Nicotine class
def ModelBuildLogisticRegression(X_train, X_test, y_train, y_test, penalty='none', class_weight=None,
                                 precision_recall_curve_=False):
    lr = LogisticRegression(class_weight=class_weight, penalty=penalty, max_iter=200)
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    if precision_recall_curve_:
        prc = []
        rec = []
        n_classes = 7
        for i in range(n_classes):
            prec, recal, _ = precision_recall_curve(y_test, y_pred, pos_label=lr.classes_[i])
            prc.append(prec)
            rec.append(recal)
        labels = y_train.unique().tolist()
        plt.figure(figsize=(10, 5))
        plt.plot(rec[0], prc[0], marker='.', label=labels[0])
        plt.plot(rec[1], prc[1], marker='.', label=labels[1])
        plt.plot(rec[2], prc[2], marker='.', label=labels[2])
        plt.plot(rec[3], prc[3], marker='.', label=labels[3])
        plt.plot(rec[4], prc[4], marker='.', label=labels[4])
        plt.plot(rec[5], prc[5], marker='.', label=labels[5])
        plt.plot(rec[6], prc[6], marker='.', label=labels[6])

        plt.title(f"Precision-Recall-Curve==>Logistic Regression(penalty={penalty}, class_weight={class_weight}) ")
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # saving figure
        plt.savefig(f"Logistic Regression(penalty={penalty}, class_weight={class_weight}).png")
        # show the plot
        plt.show()

    return y_pred


# this function check the classification report and confusion matrix
def classification_confusion_matrix(y_pred, y_test_set):
    print("\nClassification report : \n", classification_report(y_test_set, y_pred))
    cm = confusion_matrix(y_test_set, y_pred)
    print("Confusion matrix : \n", cm)


# in this function we are checking the accuracy, precision and recall on micro and macro
def modelEvaluation(y_pred, y_test_set, average='macro'):
    print(f"\nAverage : {average}")
    # Print model evaluation to predicted result
    print("Accuracy on validation set: {:.4f}".format(accuracy_score(y_test_set, y_pred)))
    print("Precision on validation set: {:.4f}".format(precision_score(y_test_set, y_pred, average=average)))
    print("Recall on validation set: {:.4f}".format(recall_score(y_test_set, y_pred, average=average)))


# This function we are checking the model on polynomial features with degree 2
def create_polynomial_model(degree, x_train, x_test, Y_train, Y_test, precision_recall_curve_=False):
    # Creates a polynomial regression model for the given degree
    poly = PolynomialFeatures(degree=degree)
    lr = LogisticRegression(max_iter=1000)
    pipe = Pipeline([('polynomial_features', poly), ('logistic_regression', lr)])
    pipe.fit(x_train, Y_train)
    y_pred = pipe.predict(x_test)

    if precision_recall_curve_:
        prc = []
        rec = []
        n_classes = 7
        for i in range(n_classes):
            prec, recal, _ = precision_recall_curve(y_test, y_pred, pos_label=pipe.classes_[i])
            prc.append(prec)
            rec.append(recal)
        labels = y_train.unique().tolist()
        plt.figure(figsize=(10, 5))
        plt.plot(rec[0], prc[0], marker='.', label=labels[0])
        plt.plot(rec[1], prc[1], marker='.', label=labels[1])
        plt.plot(rec[2], prc[2], marker='.', label=labels[2])
        plt.plot(rec[3], prc[3], marker='.', label=labels[3])
        plt.plot(rec[4], prc[4], marker='.', label=labels[4])
        plt.plot(rec[5], prc[5], marker='.', label=labels[5])
        plt.plot(rec[6], prc[6], marker='.', label=labels[6])

        plt.title(f"Precision-Recall-Curve==>Polynomial Features Logistic Regression")
        # axis labels
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        # show the legend
        plt.legend()
        # saving the figure
        plt.savefig(f"Polynomial Features Logistic Regression.png")
        # show the plot
        plt.show()
    return y_pred


if __name__ == '__main__':
    filename = 'drug_consumption.data.txt'
    df = load_dataset(filename)
    exploring_dataset(df)
    # analyse_data_histogram(df)
    df_oneHot = one_hotEncoding(df, ['Age', 'Gender', 'Education', 'Country', 'Ethnicity', 'Impulsive', 'SS'])
    X, y = extract_dependent_independent_features(df_oneHot)
    X_train, X_test, y_train, y_test = splitting_data(X, y, test_size=0.2)
    X_train_scaled, X_test_scaled = scaling_data(X_train, X_test)

    # lets check the prediction when penalty=none and class weight=None
    y_pred = ModelBuildLogisticRegression(X_train_scaled, X_test_scaled, y_train, y_test,
                                          penalty='none', class_weight=None,
                                          precision_recall_curve_=True)
    classification_confusion_matrix(y_pred, y_test)
    modelEvaluation(y_pred, y_test, average='macro')
    modelEvaluation(y_pred, y_test, average='micro')
    print("=" * 100)

    # lets check the prediction when class weight=balanced
    y_pred_balanced = ModelBuildLogisticRegression(X_train_scaled, X_test_scaled, y_train,
                                                   y_test, class_weight='balanced',
                                                   precision_recall_curve_=True)
    classification_confusion_matrix(y_pred_balanced, y_test)
    modelEvaluation(y_pred_balanced, y_test, average='macro')
    modelEvaluation(y_pred_balanced, y_test, average='micro')
    print("=" * 100)

    # lets check the prediction when penalty=l2
    y_pred_l2 = ModelBuildLogisticRegression(X_train_scaled, X_test_scaled, y_train,
                                             y_test, penalty='l2',
                                             precision_recall_curve_=True)
    classification_confusion_matrix(y_pred_l2, y_test)
    modelEvaluation(y_pred_l2, y_test, average='macro')
    modelEvaluation(y_pred_l2, y_test, average='micro')
    print("=" * 100)

    # polynomial features with 2 degree
    y_poly = create_polynomial_model(degree=2, x_train=X_train_scaled, x_test=X_test_scaled,
                                     Y_train=y_train, Y_test=y_test,
                                     precision_recall_curve_=True)
    classification_confusion_matrix(y_poly, y_test)
    modelEvaluation(y_poly, y_test, average='macro')
    modelEvaluation(y_poly, y_test, average='micro')
