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
                                max_samples=1.0, max_features=0.5,
                                bootstrap_features=False, random_state=432,
                                bootstrap=False)):
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

        # X_dsel = self.previous_X
        # y_dsel = self.previous_y
        #
        # if self.oversampled:
        #     ros = SMOTE(random_state=42)
        #     X_dsel, y_dsel = ros.fit_resample(X_dsel, y_dsel)
        #
        # if self.desMethod == "KNORAE":
        #     des = KNORAE(self.ensemble_, random_state=42)
        # elif self.desMethod == "KNORAU":
        #     des = KNORAU(self.ensemble_, random_state=42)
        # elif self.desMethod == "KNN":
        #     des = DESKNN(self.ensemble_, random_state=42)
        # elif self.desMethod == "Clustering":
        #     des = DESClustering(self.ensemble_, random_state=42)
        # else:
        #     des = KNORAE(self.ensemble_, random_state=42)
        #
        # des.fit(X_dsel, y_dsel)
        # prediction = des.predict(X)

        prediction = self._base_clf.predict(X)

        return prediction

    def score(self, X, y):
        return measure(y, self.predict(X))
