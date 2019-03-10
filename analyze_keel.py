import numpy as np
import helper as h

keel_names = open("KEEL_names.txt", "r").read().split("\n")[:-1]
clfs = h.keel_clfs()

results_hypercube_ba = np.zeros((len(keel_names), len(clfs), 10))
results_hypercube_f1 = np.zeros((len(keel_names), len(clfs), 10))
results_hypercube_gmean = np.zeros((len(keel_names), len(clfs), 10))
for i, name in enumerate(keel_names):
    name = name[:-4]
    results_ba = np.load("results/experiment_keel/%s_ba.npy" % name)
    results_f1 = np.load("results/experiment_keel/%s_f1.npy" % name)
    results_gmean = np.load("results/experiment_keel/%s_gmean.npy" % name)

    results_hypercube_ba[i] = results_ba
    results_hypercube_f1[i] = results_f1
    results_hypercube_gmean[i] = results_gmean

# print(results_hypercube_ba)
overall_ba = np.mean(results_hypercube_ba, axis=0)
overall_f1 = np.mean(results_hypercube_f1, axis=0)
overall_gmean = np.mean(results_hypercube_gmean, axis=0)
print("\n")
print(overall_ba.mean(axis=1))
print(overall_f1.mean(axis=1))
print(overall_gmean.mean(axis=1))
    # print(results_ba.mean(axis=1))
    # print(results_f1.mean(axis=1))
    # print(results_gmean.mean(axis=1))