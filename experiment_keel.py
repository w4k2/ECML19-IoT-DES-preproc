import helper as h
import numpy as np
import multiprocessing
from sklearn.model_selection import train_test_split

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
    results = np.zeros((len(clfs), 1))
    name = keel_names[i][:-4]

    for j, clfn in enumerate(clfs):
        clf = clfs[clfn]
        print(
            "Starting clf %i/%i of keel data %i/%i"
            % (j + 1, len(clfs), i + 1, len(keel_names))
        )

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5,
                                                            random_state=43)

        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
        print(score)

        print(
            "Done clf %i/%i of keel data %i/%i" % (j + 1, len(clfs), i + 1, len(keel_names))
        )

        results[j, :] = score

    np.save("results/experiment_keel/%s" % name, results)
    print(results)


jobs = []
for i, data_n in enumerate(keel_Xs):
    p = multiprocessing.Process(target=worker, args=(i, data_n))
    jobs.append(p)
    p.start()
