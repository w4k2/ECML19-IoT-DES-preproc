import csm
import matplotlib.pyplot as plt
import numpy as np
import helper as h
from tqdm import tqdm
import multiprocessing
import logging

logging.getLogger("smote_variants").setLevel(logging.CRITICAL)

# Select streams and methods
streams = h.streams()
clfs = h.clfs()

# Define worker
def worker(i, stream_n):
    """worker function"""
    stream = streams[stream_n]
    results_ba = np.zeros((len(clfs), stream.n_chunks - 1))
    results_f1 = np.zeros((len(clfs), stream.n_chunks - 1))
    results_gmean = np.zeros((len(clfs), stream.n_chunks - 1))

    for j, clfn in enumerate(clfs):
        clf = clfs[clfn]

        print(
            "Starting clf %i/%i of stream %i/%i"
            % (j + 1, len(clfs), i + 1, len(streams))
        )

        learner = csm.TestAndTrain(stream, clf)
        learner.run()

        print(
            "Done clf %i/%i of stream %i/%i" % (j + 1, len(clfs), i + 1, len(streams))
        )

        results_ba[j, :] = learner.scores_ba
        results_f1[j, :] = learner.scores_f1
        results_gmean[j, :] = learner.scores_gmean

        stream.reset()

    np.save("results/experiment_streams/%s_ba" % stream, results_ba)
    np.save("results/experiment_streams/%s_f1" % stream, results_f1)
    np.save("results/experiment_streams/%s_gmean" % stream, results_gmean)


jobs = []
for i, stream_n in enumerate(streams):
    p = multiprocessing.Process(target=worker, args=(i, stream_n))
    jobs.append(p)
    p.start()
