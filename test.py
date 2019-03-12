import helper as h
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import RepeatedStratifiedKFold
import smote_variants as sv
import numpy as np

X, y = h.datImport("cleveland-0_vs_4.dat")
rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state = 42)

# smote = SMOTE(k_neighbors=6, random_state=42)
smote1 = sv.SMOTE()
smote2 = sv.SMOTE()

# for train_index, test_index in rskf.split(X, y):
#     X_train, X_test = X[train_index], X[test_index]
#     y_train, y_test = y[train_index], y[test_index]

X1, y1= smote1.sample(X, y)
X2, y2= smote2.sample(X, y)

if np.array_equal(X1, X2):
    print("OK")
else:
    print("CHUJ")