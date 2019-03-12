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
        "Basicn": csm.Dumb(oversampler="None"),
        "KNORAEn": csm.DESlibStream(oversampler="None", desMethod="KNORAE"),
        # "KNORAUn": csm.DESlibStream(oversampler="None", desMethod="KNORAU"),
        # "KNNn": csm.DESlibStream(oversampler="None", desMethod="KNN"),
        # "Clusteringn": csm.DESlibStream(oversampler="None", desMethod="Clustering"),
        "BasicSmote": csm.Dumb(oversampler="SMOTE"),
        "KNORAESmote": csm.DESlibStream(oversampler="SMOTE", desMethod="KNORAE"),
        # "KNORAUSmote": csm.DESlibStream(oversampler="SMOTE", desMethod="KNORAU"),
        # "KNNSmote": csm.DESlibStream(oversampler="SMOTE", desMethod="KNN"),
        # "ClusteringSmote": csm.DESlibStream(oversampler="SMOTE", desMethod="Clustering"),
        # "Basicsvm": csm.Dumb(oversampler="svmSMOTE"),
        # "KNORAEsvm": csm.DESlibStream(oversampler="svmSMOTE", desMethod="KNORAE"),
        # "KNORAUsvm": csm.DESlibStream(oversampler="svmSMOTE", desMethod="KNORAU"),
        # "KNNsvm": csm.DESlibStream(oversampler="svmSMOTE", desMethod="KNN"),
        # "Clusteringsvm": csm.DESlibStream(oversampler="svmSMOTE", desMethod="Clustering"),
        # "Basicb1": csm.Dumb(oversampler="borderline1"),
        # "KNORAEb1": csm.DESlibStream(oversampler="borderline1", desMethod="KNORAE"),
        # "KNORAUb1": csm.DESlibStream(oversampler="borderline1", desMethod="KNORAU"),
        # "KNNb1": csm.DESlibStream(oversampler="borderline1", desMethod="KNN"),
        # "Clusteringb1": csm.DESlibStream(oversampler="borderline1", desMethod="Clustering"),
        # "Basicb2": csm.Dumb(oversampler="borderline2"),
        # "KNORAEb2": csm.DESlibStream(oversampler="borderline2", desMethod="KNORAE"),
        # "KNORAUb2": csm.DESlibStream(oversampler="borderline2", desMethod="KNORAU"),
        # "KNNb2": csm.DESlibStream(oversampler="borderline2", desMethod="KNN"),
        # "Clusteringb2": csm.DESlibStream(oversampler="borderline2", desMethod="Clustering"),
        # "Basicada": csm.Dumb(oversampler="Adasyn"),
        # "KNORAEada": csm.DESlibStream(oversampler="ADASYN", desMethod="KNORAE"),
        # "KNORAUada": csm.DESlibStream(oversampler="ADASYN", desMethod="KNORAU"),
        # "KNNada": csm.DESlibStream(oversampler="ADASYN", desMethod="KNN"),
        # "Clusteringada": csm.DESlibStream(oversampler="ADASYN", desMethod="Clustering"),
        # "Basicsls": csm.Dumb(oversampler="SLS"),
        # "KNORAEsls": csm.DESlibStream(oversampler="SLS", desMethod="KNORAE"),
        # "KNORAUsls": csm.DESlibStream(oversampler="SLS", desMethod="KNORAU"),
        # "KNNsls": csm.DESlibStream(oversampler="SLS", desMethod="KNN"),
        # "Clusteringsls": csm.DESlibStream(oversampler="SLS", desMethod="Clustering"),
    }

def keel_clfs():
    return {
        "Basicn": csm.DESlibKEEL2(oversampler="None", desMethod="None"),
        "KNORAEn": csm.DESlibKEEL2(oversampler="None", desMethod="KNORAE"),
        "KNORAUn": csm.DESlibKEEL2(oversampler="None", desMethod="KNORAU"),
        "KNNn": csm.DESlibKEEL2(oversampler="None", desMethod="KNN"),
        "Clusteringn": csm.DESlibKEEL2(oversampler="None", desMethod="Clustering"),
        "BasicSmote": csm.DESlibKEEL2(oversampler="SMOTE", desMethod="None"),
        "KNORAESmote": csm.DESlibKEEL2(oversampler="SMOTE", desMethod="KNORAE"),
        "KNORAUSmote": csm.DESlibKEEL2(oversampler="SMOTE", desMethod="KNORAU"),
        "KNNSmote": csm.DESlibKEEL2(oversampler="SMOTE", desMethod="KNN"),
        "ClusteringSmote": csm.DESlibKEEL2(oversampler="SMOTE", desMethod="Clustering"),
        "Basicsvm": csm.DESlibKEEL2(oversampler="svmSMOTE", desMethod="None"),
        "KNORAEsvm": csm.DESlibKEEL2(oversampler="svmSMOTE", desMethod="KNORAE"),
        "KNORAUsvm": csm.DESlibKEEL2(oversampler="svmSMOTE", desMethod="KNORAU"),
        "KNNsvm": csm.DESlibKEEL2(oversampler="svmSMOTE", desMethod="KNN"),
        "Clusteringsvm": csm.DESlibKEEL2(oversampler="svmSMOTE", desMethod="Clustering"),
        "Basicb1": csm.DESlibKEEL2(oversampler="borderline1", desMethod="None"),
        "KNORAEb1": csm.DESlibKEEL2(oversampler="borderline1", desMethod="KNORAE"),
        "KNORAUb1": csm.DESlibKEEL2(oversampler="borderline1", desMethod="KNORAU"),
        "KNNb1": csm.DESlibKEEL2(oversampler="borderline1", desMethod="KNN"),
        "Clusteringb1": csm.DESlibKEEL2(oversampler="borderline1", desMethod="Clustering"),
        "Basicb2": csm.DESlibKEEL2(oversampler="borderline2", desMethod="None"),
        "KNORAEb2": csm.DESlibKEEL2(oversampler="borderline2", desMethod="KNORAE"),
        "KNORAUb2": csm.DESlibKEEL2(oversampler="borderline2", desMethod="KNORAU"),
        "KNNb2": csm.DESlibKEEL2(oversampler="borderline2", desMethod="KNN"),
        "Clusteringb2": csm.DESlibKEEL2(oversampler="borderline2", desMethod="Clustering"),
        "Basicada": csm.DESlibKEEL2(oversampler="ADASYN", desMethod="None"),
        "KNORAEada": csm.DESlibKEEL2(oversampler="ADASYN", desMethod="KNORAE"),
        "KNORAUada": csm.DESlibKEEL2(oversampler="ADASYN", desMethod="KNORAU"),
        "KNNada": csm.DESlibKEEL2(oversampler="ADASYN", desMethod="KNN"),
        "Clusteringada": csm.DESlibKEEL2(oversampler="ADASYN", desMethod="Clustering"),
        "Basicsls": csm.DESlibKEEL2(oversampler="SLS", desMethod="None"),
        "KNORAEsls": csm.DESlibKEEL2(oversampler="SLS", desMethod="KNORAE"),
        "KNORAUsls": csm.DESlibKEEL2(oversampler="SLS", desMethod="KNORAU"),
        "KNNsls": csm.DESlibKEEL2(oversampler="SLS", desMethod="KNN"),
        "Clusteringsls": csm.DESlibKEEL2(oversampler="SLS", desMethod="Clustering"),
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
    drift_types = ["sudden", "incremental"]
    # random_states = [1337, 666, 42]
    random_states = [1337, 666]

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
                        n_drifts=5,
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
    y[y == " negative"] = 0
    y[y == " negative    "] = 0
    y[y=="positive"] = 1
    y[y == " positive"] = 1
    y=y.astype('int')
    return X, y
