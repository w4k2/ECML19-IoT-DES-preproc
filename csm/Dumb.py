"""
Dumb Delay Pool.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn import base
from sklearn import neighbors
from sklearn.metrics import f1_score, balanced_accuracy_score
from imblearn.metrics import  geometric_mean_score
import numpy as np
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN
from smote_variants import Safe_Level_SMOTE
from sklearn.tree import DecisionTreeClassifier

ba = balanced_accuracy_score
f1 = f1_score
gmean = geometric_mean_score


class Dumb(BaseEstimator, ClassifierMixin):
    """
    DumbDelayPool.

    Opis niezwykle istotnego klasyfikatora

    References
    ----------
    .. [1] A. Kowalski, B. Nowak, "Bardzo ważna praca o klasyfikatorze
    niezwykle istotnym dla przetrwania gatunku ludzkiego."

    """

    def __init__(self, ensemble_size=20, alpha=0.05, oversampler = "None"):
        """Initialization."""
        self.ensemble_size = ensemble_size
        self.alpha = alpha
        self.oversampler = oversampler

    def set_base_clf(self, base_clf=DecisionTreeClassifier()):
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

        candidate_clf = base.clone(self._base_clf)
        candidate_clf.fit(X, y)

        self.ensemble_ = [candidate_clf]

        # Return the classifier
        return self

    def remove_outliers(self, X, y):
        # Detect and remove outliers
        out_clf = neighbors.KNeighborsClassifier(n_neighbors=6)
        out_clf.fit(X, y)
        out_pp = out_clf.predict_proba(X)

        same_neighbors = (
                (out_pp[tuple([range(len(y)), y])] - (1 / out_clf.n_neighbors))
                * out_clf.n_neighbors
        ).astype(int)

        filter = same_neighbors > 3

        # What if nothing left?
        if len(np.unique(y[filter])) == 1:
            filter[np.argmax(y == 0)] = True

        return X[filter], y[filter]

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()
        X, y = check_X_y(X, y)

        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes
            self.ensemble_ = []

        self.X_, self.y_ = X, y

        train_X, train_y = self.remove_outliers(X, y)
        # train_X, train_y = X, y

        unique, counts = np.unique(train_y, return_counts=True)

        k_neighbors = 5
        if counts[0] - 1 < 5:
            k_neighbors = counts[0] - 1

        if self.oversampler == "SMOTE" and k_neighbors > 0:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            train_X, train_y = smote.fit_resample(train_X, train_y)
        elif self.oversampler == "svmSMOTE" and k_neighbors > 0:
            try:
                svmSmote = SVMSMOTE(random_state=42, k_neighbors=k_neighbors)
                train_X, train_y = svmSmote.fit_resample(train_X, train_y)
            except ValueError:
                pass
        elif self.oversampler == "borderline1" and k_neighbors > 0:
            borderlineSmote1 = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors, kind='borderline-1')
            train_X, train_y = borderlineSmote1.fit_resample(train_X, train_y)
        elif self.oversampler == "borderline2" and k_neighbors > 0:
            borderlineSmote2 = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors, kind='borderline-2')
            train_X, train_y = borderlineSmote2.fit_resample(train_X, train_y)
        elif self.oversampler == "ADASYN" and k_neighbors > 0:
            try:
                adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
                train_X, train_y = adasyn.fit_resample(train_X, train_y)
            except RuntimeError:
                pass
        elif self.oversampler == "SLS" and k_neighbors > 0:
            sls = Safe_Level_SMOTE(n_neighbors=k_neighbors)
            train_X, train_y = sls.sample(train_X, train_y)

        # Testing all models
        scores = np.array([ba(y, clf.predict(X)) for clf in self.ensemble_])

        # Pruning
        if len(self.ensemble_) > 1:
            alpha_good = scores > (0.5 + self.alpha)
            # print(scores)
            self.ensemble_ = [self.ensemble_[i] for i in np.where(alpha_good)[0]]

        if len(self.ensemble_) > self.ensemble_size - 1:
            worst = np.argmin(scores)
            del self.ensemble_[worst]

        # Preparing and training new candidate
        self.ensemble_.append(base.clone(self._base_clf).fit(train_X, train_y))

        # print("ENSEMBLE OF ", len(self.ensemble_))

        # print(len(self.ensemble_))

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict(self, X):
        """Hard decision."""
        # print("PREDICT")
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        esm = self.ensemble_support_matrix(X)
        average_support = np.mean(esm, axis=0)
        prediction = np.argmax(average_support, axis=1)

        return prediction

    def score(self, X, y):
        return ba(y, self.predict(X)), f1(y, self.predict(X)), gmean(y, self.predict(X))
