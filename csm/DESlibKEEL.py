"""
Dumb Delay Pool.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn import base
from sklearn.ensemble import BaggingClassifier
from sklearn import neighbors
from sklearn.metrics import f1_score, balanced_accuracy_score
import numpy as np
import matplotlib.pyplot as plt
from deslib.des import KNORAE, KNORAU, DESKNN, DESClustering
from deslib.dcs import Rank, LCA
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN
from scipy.spatial import distance
from sklearn.tree import DecisionTreeClassifier

measure = balanced_accuracy_score


class DESlibKEEL(BaseEstimator, ClassifierMixin):
    """
        note
    """

    def __init__(
        self, ensemble_size=3, desMethod="KNORAE", oversampled=True
    ):
        """Initialization."""
        self.ensemble_size = ensemble_size
        self.desMethod = desMethod
        self.oversampled = oversampled

    def set_base_clf(self, base_clf=BaggingClassifier(
                                neighbors.KNeighborsClassifier(),
                                n_estimators=10,
                                max_samples=0.5, max_features=1.0,
                                bootstrap_features=False, random_state=42,
                                bootstrap=True)):
        """Establish base classifier."""
        self._base_clf = base_clf

    # Fitting
    def fit(self, X, y):
        """Fitting."""
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        self.X_dsel = X
        self.y_dsel = y

        if self.oversampled:
            ros = SMOTE(random_state=42)
            X, y = ros.fit_resample(X, y)

        self._base_clf.fit(X, y)
        # Return the classifier
        return self

    def predict(self, X):
        """Hard decision."""
        # print("PREDICT")
        # Check is fit had been called
        # check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        if self.oversampled:
            ros = SMOTE(random_state=42)
            self.X_dsel, self.y_dsel = ros.fit_resample(self.X_dsel, self.y_dsel)

        if self.desMethod == "KNORAE":
            des = KNORAE(self._base_clf.estimators_, random_state=42)
            des.fit(self.X_, self.y_)
            prediction = des.predict(X)
        elif self.desMethod == "KNORAU":
            des = KNORAU(self._base_clf.estimators_, random_state=42)
            des.fit(self.X_dsel, self.y_dsel)
            prediction = des.predict(X)
        elif self.desMethod == "KNN":
            des = DESKNN(self._base_clf.estimators_, random_state=42)
            des.fit(self.X_dsel, self.y_dsel)
            prediction = des.predict(X)
        elif self.desMethod == "Clustering":
            des = DESClustering(self._base_clf.estimators_, random_state=42)
            des.fit(self.X_dsel, self.y_dsel)
            prediction = des.predict(X)
        else:
            prediction = self._base_clf.predict(X)


        return prediction

    def score(self, X, y):
        return measure(y, self.predict(X))
