from RBO import RBO
from imblearn.over_sampling import RandomOverSampler, SMOTE, SVMSMOTE, BorderlineSMOTE, ADASYN

import helper as h

smote = SMOTE(random_state=42)
rand = RandomOverSampler(random_state=42)

X, y = h.datImport("ecoli4.dat")

print(X.shape)
print(y.shape)

rbo = RBO()
# X, y = rbo.fit_sample(X, y)
X, y = smote.fit_resample(X, y)

print(X.shape)
print(y.shape)