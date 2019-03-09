import csm
import numpy as np
import pandas as pd
from scipy.stats import ranksums

p = 0.05


# def clfs():
#     return {
#         "MDE": csm.MDE(),
#         "KNORAE": csm.DESlibStream(desMethod="KNORAE"),
#         "KNORAU": csm.DESlibStream(desMethod="KNORAU"),
#         "Rank": csm.DESlibStream(desMethod="Rank"),
#         "LCA": csm.DESlibStream(desMethod="LCA"),
#     }


# def clfs_nos():
#     return {
#         "MDE": csm.MDE(),
#         "KNORAE": csm.DESlibStream(desMethod="KNORAE", oversampled=False),
#         "KNORAU": csm.DESlibStream(desMethod="KNORAU", oversampled=False),
#         "Rank": csm.DESlibStream(desMethod="Rank", oversampled=False),
#         "LCA": csm.DESlibStream(desMethod="LCA", oversampled=False),
#     }

def clfs():
    return {
        # "Basic": csm.Dumb(ensemble_size=2),
        "KNNov": csm.DESlibStream(desMethod="KNN", oversampled=True, ensemble_size=3),
        "KNN": csm.DESlibStream(desMethod="KNN", oversampled=False, ensemble_size=3),
        # "Clustering": csm.DESlibStream(desMethod="Clustering", oversampled=False, ensemble_size=20),
        # "KNORAE": csm.DESlibStream(desMethod="KNORAE", oversampled=False, ensemble_size=20),
        # "KNORAU": csm.DESlibStream(desMethod="KNORAU", oversampled=False, ensemble_size=20),
        # "KNNov": csm.DESlibStream(desMethod="KNN", oversampled=True, ensemble_size=20),
        # "Clusteringov": csm.DESlibStream(desMethod="Clustering", oversampled=True, ensemble_size=20),
        # "KNORAEov": csm.DESlibStream(desMethod="KNORAE", oversampled=True, ensemble_size=20),
        # "KNORAUov": csm.DESlibStream(desMethod="KNORAU", oversampled=True, ensemble_size=20),
    }


def real_streams():
    streams = ["elecNormNew"]
    return streams


def streams():
    # Variables
    # distributions = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    distributions = [[0.1, 0.9]]
    # label_noises = [0.0, 0.1, 0.2, 0.3]
    label_noises = [0.0]
    # drift_types = ["incremental", "sudden"]
    drift_types = ["sudden"]
    # random_states = [1337, 666, 42]
    random_states = [1337]

    # Prepare streams
    streams = {}
    for drift_type in drift_types:
        for distribution in distributions:
            for random_state in random_states:
                for flip_y in label_noises:
                    stream = csm.StreamGenerator(
                        drift_type=drift_type,
                        distribution=distribution,
                        random_state=random_state,
                        flip_y=flip_y,
                        n_drifts=1,
                    )
                    streams.update({str(stream): stream})

    return streams


def tabrow(what, res):
    mean = np.mean(res, axis=1)
    std = np.std(res, axis=1)

    width = len(mean)

    leader = np.argmax(mean)

    pvalues = np.array([ranksums(res[leader], res[i]).pvalue for i in range(width)])
    dependences = pvalues > p

    return (
        ("\\emph{%s}" % what)
        + " & "
        + (
            " & ".join(
                [
                    "%s %.3f" % ("\\bfseries" if dependences[i] else "", mean[i])
                    for i in range(width)
                ]
            )
            + " \\\\"
        )
    )

def datImport(filename):
    df = pd.read_csv('KEEL_data/' + filename, header=None, sep=",", comment="@")
    array = df.values
    X = array[:,:-1]
    y = array[:,-1:]
    y = np.reshape(y, (y.shape[0],))
    y[y=="negative"] = 0
    y[y=="positive"] = 1
    return X, y
