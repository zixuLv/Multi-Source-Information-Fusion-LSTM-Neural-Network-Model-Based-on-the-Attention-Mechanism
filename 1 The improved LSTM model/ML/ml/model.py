# model.py
# -*- coding: utf-8 -*-

import time
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
# from scikeras.wrappers import KerasClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
import numpy as np


def Get_Model(which, n_inputs=None):
    """
    Return a machine learning model based on the specified name.
    """
    if which == "RandomForest":
        return RandomForestClassifier(max_depth=10, random_state=0)
    elif which == "SupportVectorMachine":
        return SVC(verbose=True)
    elif which == "BayesClassifier":
        return GaussianNB()
    elif which == "DecisionTree":
        return DecisionTreeClassifier()
    elif which == "KNN":
        return KNeighborsClassifier()
    elif which == "Multi_LayerPerceptron":
        return MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64, 32), max_iter=500)
    elif which == "Logistic_Regression":
        return LogisticRegression(penalty='l2', max_iter=1000)
    elif which == "LinearDiscriminant":
        return LinearDiscriminantAnalysis()
    elif which == "GBDT":
        return GradientBoostingClassifier(n_estimators=200)
    elif which == "QDA":
        return QuadraticDiscriminantAnalysis()
    else:
        return AdaBoostClassifier()


def train(x, y, x_test, f_train, f_test, which):
    """
    Train the selected model and return predictions on training and test sets.
    """
    st = time.time()

    # If x and x_test are lists, convert them to NumPy arrays
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(x_test, list):
        x_test = np.array(x_test)

    if which in ["CNN", "LSTM"]:
        n_features = x.shape[1]

        # For CNN and LSTM models, reshape data to match required input format
        x = x.reshape((x.shape[0], x.shape[1], 1))
        x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

        Model = Get_Model(which, n_inputs=n_features)
    else:
        Model = Get_Model(which)

    Model.fit(x, y)

    ed = time.time() - st  # Training time

    y_predict = Model.predict(x)
    y_test_predict = Model.predict(x_test)

    return y_predict, y_test_predict