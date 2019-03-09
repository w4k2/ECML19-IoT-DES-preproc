import helper as h
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN

keel_data = open("KEEL_names.txt", "r").read().split("\n")[:-1]


# X, y = h.datImport("yeast-2_vs_4.dat")
#
# ros = SMOTE(random_state=42)
# train_X, train_y = ros.fit_resample(X, y)
#
#
# print(X)
# print(y)
# print(train_y)