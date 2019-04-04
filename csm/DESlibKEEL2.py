"""
Dumb Delay Pool.
"""
import csm
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.utils import resample
from sklearn.ensemble import BaggingClassifier
from sklearn import neighbors, base
from sklearn.metrics import f1_score, balanced_accuracy_score
from deslib.des import KNORAE, KNORAU, DESKNN, DESClustering
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN
from imblearn.metrics  import geometric_mean_score
from smote_variants import Safe_Level_SMOTE
import smote_variants as sv
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedShuffleSplit

ba = balanced_accuracy_score
f1 = f1_score
gmean = geometric_mean_score


class DESlibKEEL2(BaseEstimator, ClassifierMixin):
    """
        note
    """

    def __init__(
        self, ensemble_size=20, desMethod="KNORAE", oversampler="SMOTE"
    ):
        """Initialization."""
        self.ensemble_size = ensemble_size
        self.desMethod = desMethod
        self.oversampler = oversampler
        self.ensemble_ = []

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

        self.X_dsel = X
        self.y_dsel = y

        unique, counts = np.unique(self.y_dsel, return_counts=True)
        k_neighbors = 5
        minority_n = counts[1]
        if counts[1] - 1 < 5:
            print("TAKKKKK")
            k_neighbors = counts[1] - 1

        if self.oversampler == "SMOTE" and k_neighbors > 0:
            smote = SMOTE(random_state=42, k_neighbors=k_neighbors)
            self.X_dsel, self.y_dsel = smote.fit_resample(self.X_dsel, self.y_dsel)
        elif self.oversampler == "svmSMOTE" and k_neighbors > 0:
            try:
                svmSmote = SVMSMOTE(random_state=42, k_neighbors=k_neighbors)
                self.X_dsel, self.y_dsel = svmSmote.fit_resample(self.X_dsel, self.y_dsel)
            except ValueError:
                pass
        elif self.oversampler == "borderline1" and k_neighbors > 0:
            borderlineSmote1 = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors, kind='borderline-1')
            self.X_dsel, self.y_dsel = borderlineSmote1.fit_resample(self.X_dsel, self.y_dsel)
        elif self.oversampler == "borderline2" and k_neighbors > 0:
            borderlineSmote2 = BorderlineSMOTE(random_state=42, k_neighbors=k_neighbors, kind='borderline-2')
            self.X_dsel, self.y_dsel = borderlineSmote2.fit_resample(self.X_dsel, self.y_dsel)
        elif self.oversampler == "ADASYN" and k_neighbors > 0:
            try:
                adasyn = ADASYN(random_state=42, n_neighbors=k_neighbors)
                self.X_dsel, self.y_dsel = adasyn.fit_resample(self.X_dsel, self.y_dsel)
            except RuntimeError:
                pass
        elif self.oversampler == "SLS" and k_neighbors > 0:
            sls = Safe_Level_SMOTE(n_neighbors=k_neighbors)
            self.X_dsel, self.y_dsel = sls.sample(self.X_dsel, self.y_dsel)

        for i in range(1, self.ensemble_size):
            # all minority in each bootstrap
            bootstrap_size = 1.0 * X.shape[0] - minority_n
            random_state = 42 + i
            X_bag, y_bag = resample(X[y == 0], y[y == 0], n_samples=int(bootstrap_size), random_state=random_state, replace=True)
            y_bag = np.append(y_bag, y[y == 1])
            X_bag = np.append(X_bag, X[y == 1], axis=0)

            # stratified bagging
            # bootstrap_size = 1.0 * X.shape[0]
            # X_bag, y_bag = resample(X, y, n_samples=int(bootstrap_size), random_state=random_state, replace=True)
            #
            # unique, counts = np.unique(y_bag, return_counts=True)
            # pow = 2;
            # while unique.shape == (1,):
            #     X_bag, y_bag = resample(X, y, n_samples=int(bootstrap_size), random_state=random_state**pow, replace=True)
            #     pow = pow + 1
            #     unique, counts = np.unique(y_bag, return_counts=True)

            k_neighbors = 5
            if counts[1] - 1 < 5:
                k_neighbors = counts[1] - 1
                print("TAKKKKK")

            if self.oversampler == "SMOTE" and k_neighbors > 0:
                smote = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
                X_bag, y_bag = smote.fit_resample(X_bag, y_bag)
            elif self.oversampler == "svmSMOTE" and k_neighbors > 0:
                try:
                    svmSmote = SVMSMOTE(random_state=random_state, k_neighbors=k_neighbors)
                    X_bag, y_bag = svmSmote.fit_resample(X_bag, y_bag)
                except ValueError:
                    pass
            elif self.oversampler == "borderline1" and k_neighbors > 0:
                borderlineSmote1 = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors, kind='borderline-1')
                X_bag, y_bag = borderlineSmote1.fit_resample(X_bag, y_bag)
            elif self.oversampler == "borderline2" and k_neighbors > 0:
                borderlineSmote2 = BorderlineSMOTE(random_state=random_state, k_neighbors=k_neighbors, kind='borderline-2')
                X_bag, y_bag = borderlineSmote2.fit_resample(X_bag, y_bag)
            elif self.oversampler == "ADASYN" and k_neighbors > 0:
                try:
                    adasyn = ADASYN(random_state=random_state, n_neighbors=k_neighbors)
                    X_bag, y_bag = adasyn.fit_resample(X_bag, y_bag)
                except RuntimeError:
                    pass
            elif self.oversampler == "SLS" and k_neighbors > 0:
                sls = Safe_Level_SMOTE(n_neighbors=k_neighbors)
                X_bag, y_bag = sls.sample(X_bag, y_bag)

            self.ensemble_.append(base.clone(self._base_clf).fit(X_bag, y_bag))

        return self

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([clf.predict_proba(X) for clf in self.ensemble_])

    def predict(self, X):
        """Hard decision."""
        # print("PREDICT")
        # Check is fit had been called
        # check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        if self.desMethod == "KNORAE":
            des = KNORAE(self.ensemble_, random_state=42)
            des.fit(self.X_dsel, self.y_dsel)
            prediction = des.predict(X)
        elif self.desMethod == "KNORAU":
            des = KNORAU(self.ensemble_, random_state=42)
            des.fit(self.X_dsel, self.y_dsel)
            prediction = des.predict(X)
        elif self.desMethod == "KNN":
            des = DESKNN(self.ensemble_, random_state=42)
            des.fit(self.X_dsel, self.y_dsel)
            prediction = des.predict(X)
        elif self.desMethod == "Clustering":
            des = DESClustering(self.ensemble_, random_state=42)
            des.fit(self.X_dsel, self.y_dsel)
            prediction = des.predict(X)
        else:
            esm = self.ensemble_support_matrix(X)
            average_support = np.mean(esm, axis=0)
            prediction = np.argmax(average_support, axis=1)


        return prediction

    def score(self, X, y):
        return ba(y, self.predict(X)), f1(y, self.predict(X)), gmean(y, self.predict(X))
