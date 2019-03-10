import helper as h
import numpy as np
import multiprocessing
from sklearn.model_selection import RepeatedStratifiedKFold, train_test_split

from sklearn.ensemble import BaggingClassifier
from sklearn import neighbors

# prepare data
keel_names = open("KEEL_names.txt", "r").read().split("\n")[:-1]
keel_Xs = {}
keel_ys = {}

clfs = h.keel_clfs()

for name in keel_names:
    X, y = h.datImport(name)
    keel_Xs.update({name: X})
    keel_ys.update({name: y})

# Define worker
def worker(i, data_n):
    """worker function"""
    X = keel_Xs[data_n]
    y = keel_ys[data_n]
    results_ba = np.zeros((len(clfs), 10))
    results_f1 = np.zeros((len(clfs), 10))
    results_gmean = np.zeros((len(clfs), 10))
    name = keel_names[i][:-4]

    for j, clfn in enumerate(clfs):
        clf = clfs[clfn]
        print(
            "Starting clf %i/%i of keel data %i/%i"
            % (j + 1, len(clfs), i + 1, len(keel_names))
        )

        # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
        #                                                     random_state=42)

        rskf = RepeatedStratifiedKFold(n_splits=2, n_repeats=5, random_state = 42)

        scores_ba = []
        scores_f1 = []
        scores_gmean = []
        for train_index, test_index in rskf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            clf.fit(X_train, y_train)
            score_ba, score_f1, score_gmean = clf.score(X_test, y_test)
            scores_ba.append(score_ba)
            scores_f1.append(score_f1)
            scores_gmean.append(score_gmean)

        print(
            "Done clf %i/%i of keel data %i/%i" % (j + 1, len(clfs), i + 1, len(keel_names))
        )
        results_ba[j, :] = scores_ba
        results_f1[j, :] = scores_f1
        results_gmean[j, :] = scores_gmean

    np.save("results/experiment_keel/%s_ba" % name, results_ba)
    np.save("results/experiment_keel/%s_f1" % name, results_f1)
    np.save("results/experiment_keel/%s_gmean" % name, results_gmean)


jobs = []
for i, data_n in enumerate(keel_Xs):
    p = multiprocessing.Process(target=worker, args=(i, data_n))
    jobs.append(p)
    p.start()
