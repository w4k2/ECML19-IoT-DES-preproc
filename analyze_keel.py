import numpy as np

keel_names = open("KEEL_names.txt", "r").read().split("\n")[:-1]

for name in keel_names:
    name = name[:-4]
    results = np.load("results/experiment_keel/%s.npy" % name)
    # print(results)
    print(results.mean(axis=1))