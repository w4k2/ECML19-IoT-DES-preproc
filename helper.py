import csm
import numpy as np
from scipy.stats import wilcoxon
import string

p = 0.05


def clfs():
    return {
        "Naive-None": csm.Dumb(oversampler="None"),
        "KNORAE-None": csm.DESlibStream(oversampler="None", desMethod="KNORAE"),
        "KNORAU-None": csm.DESlibStream(oversampler="None", desMethod="KNORAU"),
        "KNN-None": csm.DESlibStream(oversampler="None", desMethod="KNN"),
        "ClusteringNone": csm.DESlibStream(oversampler="None", desMethod="Clustering"),
        "Naive-Smote": csm.Dumb(oversampler="SMOTE"),
        "KNORAE-Smote": csm.DESlibStream(oversampler="SMOTE", desMethod="KNORAE"),
        "KNORAUSmote": csm.DESlibStream(oversampler="SMOTE", desMethod="KNORAU"),
        "KNN-Smote": csm.DESlibStream(oversampler="SMOTE", desMethod="KNN"),
        "ClusteringSmote": csm.DESlibStream(oversampler="SMOTE", desMethod="Clustering"),
        "Naive-SVM": csm.Dumb(oversampler="svmSMOTE"),
        "KNORAE-SVM": csm.DESlibStream(oversampler="svmSMOTE", desMethod="KNORAE"),
        "KNORAU-SVM": csm.DESlibStream(oversampler="svmSMOTE", desMethod="KNORAU"),
        "KNN-SVM": csm.DESlibStream(oversampler="svmSMOTE", desMethod="KNN"),
        "Clusteringsvm": csm.DESlibStream(oversampler="svmSMOTE", desMethod="Clustering"),
        "Naive-B1": csm.Dumb(oversampler="borderline1"),
        "KNORAEb1": csm.DESlibStream(oversampler="borderline1", desMethod="KNORAE"),
        "KNORAUb1": csm.DESlibStream(oversampler="borderline1", desMethod="KNORAU"),
        "KNN-B1": csm.DESlibStream(oversampler="borderline1", desMethod="KNN"),
        "Clusteringb1": csm.DESlibStream(oversampler="borderline1", desMethod="Clustering"),
        "Naive-B2": csm.Dumb(oversampler="borderline2"),
        "KNORAE-B2": csm.DESlibStream(oversampler="borderline2", desMethod="KNORAE"),
        "KNORAU-B2": csm.DESlibStream(oversampler="borderline2", desMethod="KNORAU"),
        "KNN-B2": csm.DESlibStream(oversampler="borderline2", desMethod="KNN"),
        "ClusteringB2": csm.DESlibStream(oversampler="borderline2", desMethod="Clustering"),
        "Naive-ADASYN": csm.Dumb(oversampler="Adasyn"),
        "KNORAEada": csm.DESlibStream(oversampler="ADASYN", desMethod="KNORAE"),
        "KNORAUada": csm.DESlibStream(oversampler="ADASYN", desMethod="KNORAU"),
        "KNN-ADASYN": csm.DESlibStream(oversampler="ADASYN", desMethod="KNN"),
        "Clusteringada": csm.DESlibStream(oversampler="ADASYN", desMethod="Clustering"),
        "Basicsls": csm.Dumb(oversampler="SLS"),
        "KNORAEsls": csm.DESlibStream(oversampler="SLS", desMethod="KNORAE"),
        "KNORAUsls": csm.DESlibStream(oversampler="SLS", desMethod="KNORAU"),
        "KNNsls": csm.DESlibStream(oversampler="SLS", desMethod="KNN"),
        "Clusteringsls": csm.DESlibStream(oversampler="SLS", desMethod="Clustering"),
    }


def streams():
    # Variables
    distributions = [[0.1, 0.9], [0.2, 0.8], [0.3, 0.7], [0.4, 0.6]]
    label_noises = [0.0, 0.1, 0.2]
    drift_types = ["sudden", "incremental"]
    random_states = [522, 825, 37]

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


def tabrow_indices(what, res):

    mean = np.mean(res, axis=1)

    width = len(mean)
    leader = np.argmax(mean)


    pvaluesm = np.array([wilcoxon(res[leader], res[i], zero_method="zsplit").pvalue for i in range(width)])
    dependencesm = pvaluesm > p

    pvalues = np.zeros(shape=(width, width))

    for i in range(width):
        for j in range(width):
            pvalues[i][j] = wilcoxon(res[i], res[j], zero_method="zsplit").pvalue
    dependences = pvalues < p


    for i in range(width):
        for j in range(width):
            if mean[i] > mean[j] and dependences[i][j]==True:
                dependences[i][j] = True
            else:
                dependences[i][j] = False


    for i in range(width):
        if np.argwhere(dependences[i] == True).shape[0] == width-1:
            print("$_{All}$", end=" & ")
        elif np.argwhere(dependences[i]==True).shape[0]!=0:
            array = (np.argwhere(dependences[i] == True).reshape(-1, np.argwhere(dependences[i] == True).shape[0])) + 1
            if width == 7:
                str_array = [[string.ascii_lowercase[element - 1] for element in row] for row in array]
                array = str_array
            for row in array:
                print("$_{", ",".join(map(str, row)), "}$", end=" & ")
        else:
            print("--", end=" & ")

    return (
        ("\\emph{%s}" % what)
        + " & "
        + (
            " & ".join(
                [
                    "%s %.3f" % ("\\bfseries" if dependencesm[i] else "", mean[i])
                    for i in range(width)
                ]
            )
            + " \\\\"
        )
    )
