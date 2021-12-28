import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np
from matplotlib import rcParams


rcParams["font.family"] = "monospace"

# Dane

df10b = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.650, 0.664, 0.677, 0.657, 0.681, 0.651, 0.649],
    'KNORA-E': [0.717, 0.741, 0.751, 0.741, 0.738, 0.718,  0.738],
    'KNORA-U': [0.729,  0.768, 0.771, 0.763, 0.772, 0.740, 0.768],
    'DES-KNN': [0.743, 0.762, 0.770, 0.762, 0.763, 0.741, 0.758],
    'DES-CLustering': [0.725, 0.754, 0.762, 0.750, 0.755, 0.728, 0.752]
})

df10g = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.544, 0.569, 0.591, 0.555, 0.598, 0.544, 0.542],
    'KNORA-E': [0.683, 0.729, 0.735, 0.724, 0.729, 0.702, 0.729],
    'KNORA-U': [0.676, 0.742, 0.742, 0.734, 0.751, 0.705, 0.745],
    'DES-KNN': [0.705, 0.748, 0.752, 0.744, 0.752,  0.723, 0.748],
    'DES-CLustering': [0.679, 0.733, 0.738, 0.726, 0.740, 0.702, 0.734]
})

df20b = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.744, 0.757, 0.771, 0.754, 0.773, 0.747, 0.744],
    'KNORA-E': [0.779, 0.793, 0.801, 0.793, 0.782, 0.776, 0.788],
    'KNORA-U': [0.809, 0.829, 0.833, 0.829, 0.830, 0.819, 0.830],
    'DES-KNN': [0.814, 0.820, 0.826, 0.820, 0.814, 0.805, 0.814],
    'DES-CLustering': [0.800, 0.815, 0.820, 0.813, 0.811, 0.800, 0.813]
})

df20g = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.704, 0.724, 0.744, 0.719, 0.746, 0.708, 0.704],
    'KNORA-E': [0.768, 0.789, 0.797, 0.789, 0.780, 0.772, 0.786],
    'KNORA-U': [0.792, 0.820, 0.825, 0.821, 0.825, 0.809, 0.822],
    'DES-KNN': [0.803, 0.816, 0.822, 0.817, 0.812, 0.802, 0.811],
    'DES-CLustering': [0.783, 0.807, 0.813, 0.805, 0.806, 0.792, 0.807]
})

df30b = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.800, 0.806, 0.816, 0.808, 0.819, 0.802, 0.800],
    'KNORA-E': [0.806, 0.815, 0.819, 0.813, 0.800, 0.801, 0.809],
    'KNORA-U': [0.846, 0.856, 0.858, 0.856, 0.855, 0.850, 0.856],
    'DES-KNN': [0.844, 0.846, 0.847, 0.844, 0.836, 0.833, 0.838],
    'DES-CLustering': [0.834, 0.841, 0.843, 0.839, 0.835, 0.831, 0.839]
})

df30g = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.786, 0.794, 0.807, 0.797, 0.810, 0.790, 0.786],
    'KNORA-E': [0.803, 0.814, 0.817, 0.811, 0.799, 0.799, 0.808],
    'KNORA-U': [0.840, 0.852, 0.855, 0.853, 0.853, 0.847, 0.853],
    'DES-KNN': [0.841, 0.844, 0.846, 0.843, 0.835, 0.831, 0.837],
    'DES-CLustering': [0.828, 0.838, 0.841, 0.836, 0.833, 0.828, 0.836]
})

df40b = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.827, 0.828, 0.834, 0.832, 0.836, 0.827, 0.827],
    'KNORA-E': [0.819, 0.823, 0.823, 0.821, 0.811, 0.815, 0.818],
    'KNORA-U': [0.864, 0.867, 0.868, 0.867, 0.866, 0.864, 0.868],
    'DES-KNN': [0.857, 0.856, 0.856, 0.854, 0.848, 0.849, 0.852],
    'DES-CLustering': [0.851, 0.853, 0.853, 0.852, 0.848, 0.847, 0.852]
})

df40g = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.822, 0.824, 0.831, 0.828, 0.833, 0.822, 0.822],
    'KNORA-E': [0.818, 0.822, 0.823, 0.820, 0.810, 0.814, 0.817],
    'KNORA-U': [0.862, 0.856, 0.867, 0.866, 0.865, 0.863, 0.866],
    'DES-KNN': [0.856, 0.855, 0.854, 0.853, 0.847, 0.848, 0.851],
    'DES-CLustering': [0.849, 0.851, 0.852, 0.850, 0.847, 0.845, 0.850]
})

dfsudb = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.717, 0.732, 0.746, 0.727, 0.749, 0.721, 0.716],
    'KNORA-E': [0.756, 0.771, 0.780, 0.771, 0.763, 0.753, 0.767],
    'KNORA-U': [0.780, 0.803, 0.807, 0.801, 0.805, 0.792, 0.802],
    'DES-KNN': [0.784, 0.793, 0.800, 0.794, 0.789, 0.776, 0.788],
    'DES-CLustering': [0.774, 0.790, 0.797, 0.788, 0.786, 0.773, 0.787]
})

dfsudg = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.657, 0.679, 0.700, 0.672, 0.706, 0.662, 0.656],
    'KNORA-E': [0.735, 0.764, 0.771, 0.761, 0.758, 0.745, 0.761],
    'KNORA-U': [0.750, 0.787, 0.790, 0.783, 0.793, 0.775, 0.788],
    'DES-KNN': [0.762, 0.784, 0.789, 0.783, 0.783, 0.767, 0.781],
    'DES-CLustering': [0.748, 0.777, 0.783, 0.773, 0.778, 0.761, 0.776]
})

dfincb = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.677, 0.689, 0.703, 0.684, 0.704, 0.677, 0.676],
    'KNORA-E': [0.741, 0.762, 0.771, 0.762, 0.757, 0.741, 0.759],
    'KNORA-U': [0.757, 0.793, 0.796, 0.791, 0.797, 0.767, 0.795],
    'DES-KNN': [0.773, 0.788, 0.796, 0.789, 0.789, 0.770, 0.784],
    'DES-CLustering': [0.751, 0.778, 0.785, 0.775, 0.780, 0.756, 0.778]
})

dfincg = pd.DataFrame({
    'group': ['None', 'SMOTE', 'SVM-SMOTE', 'B1-SMOTE', 'B2-SMOTE', 'SL-SMOTE', 'ADASYN'],
    'Naive': [0.592, 0.613, 0.635, 0.602, 0.638, 0.591, 0.590],
    'KNORA-E': [0.717, 0.754, 0.761, 0.752, 0.751, 0.729, 0.753],
    'KNORA-U': [0.718, 0.775, 0.777, 0.772, 0.783, 0.739, 0.779],
    'DES-KNN': [0.746, 0.780, 0.784, 0.778, 0.782, 0.757, 0.779],
    'DES-CLustering': [0.714, 0.763, 0.768, 0.758, 0.769, 0.733, 0.766]
})


dfs = [df10b, df10g, df20b, df20g, df30b, df30g, df40b, df40g, dfsudb, dfsudg, dfincb, dfincg]
# tytuły i nazwy plików
names = ["10% of minority class - BAC", "10% of minority class - G-mean", "20% of minority class - BAC", "20% of minority class - G-mean","30% of minority class - BAC", "30% of minority class - G-mean","40% of minority class - BAC", "40% of minority class - G-mean", "Sudden drift - BAC", "Sudden drift - G-mean", "Incremental drift - BAC", "Incremental drift - G-mean"]

files = ["radar10b", "radar10g", "radar20b", "radar20g", "radar30b", "radar30g", "radar40b", "radar40g", "radarsudb", "radarsudg", "radardfincb", "radardfincg"]

# kolory
colors = [(0, 0, 0), (0, 0, 0.9), (0.9, 0, 0), (0.9, 0, 0), (0, 0, 0.9), (0, 0, 0.9), (0.9, 0, 0)]
# styl linii
ls = ["-", "-", "-", "--", "--", ":", ":"]


colors = ["black", "blue", "blue", "red", "red", "blue", "green"]
ls = ["-", "-", "--", "-", "--", ":", "-"]


# grubosc linii
lw = [1, 1, 1, 1, 1, 1, 1]
t = 0
for i, title in enumerate(names):

    # number of variable
    df = dfs[i]
    categories = list(df)[1:]
    N = len(categories)

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)

    # If you want the first axis to be on top:
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)

    # No shitty border
    ax.spines["polar"].set_visible(False)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, categories)

    # Adding plots
    for i in range(7):
        values = df.loc[i].drop("group").values.flatten().tolist()
        values += values[:1]
        print(values)
        values = [float(i) for i in values]
        ax.plot(
            angles, values, label=df.iloc[i, 0], c=colors[i], ls=ls[i], lw=lw[i],
        )

    # Add legend
    plt.legend(
        loc="lower center",
        ncol=4,
        columnspacing=1,
        frameon=False,
        bbox_to_anchor=(0.5, -0.15),
        fontsize=6,
    )

    # Add a grid
    plt.grid(ls=":", c=(0.7, 0.7, 0.7))

    # Add a title
    plt.title("%s" % (title), size=8, y=1.08, fontfamily="serif")
    plt.tight_layout()

    # Draw labels
    a = np.linspace(0, 1, 8)
    plt.yticks(
            [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            ["0.5", "0.6", "0.7", "0.8", "0.9", "1.0"],
            fontsize=6,
        )
    plt.ylim(0.5, 1.0)
    plt.gcf().set_size_inches(4, 3.5)
    plt.gcf().canvas.draw()
    angles = np.rad2deg(angles)

    ax.set_rlabel_position((angles[0] + angles[1]) / 2)

    har = [(a >= 90) * (a <= 270) for a in angles]

    for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
        x, y = label.get_position()
        print(label, angle)
        lab = ax.text(
            x, y, label.get_text(), transform=label.get_transform(), fontsize=6,
        )
        lab.set_rotation(angle)

        if har[z]:
            lab.set_rotation(180 - angle)
        else:
            lab.set_rotation(-angle)
        lab.set_verticalalignment("center")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")

    for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
        x, y = label.get_position()
        print(label, angle)
        lab = ax.text(
            x,
            y,
            label.get_text(),
            transform=label.get_transform(),
            fontsize=4,
            c=(0.7, 0.7, 0.7),
        )
        lab.set_rotation(-(angles[0] + angles[1]) / 2)

        lab.set_verticalalignment("bottom")
        lab.set_horizontalalignment("center")
        lab.set_rotation_mode("anchor")

    ax.set_xticklabels([])
    ax.set_yticklabels([])

    plt.savefig("figures/radars/%s.eps" % (files[t]), bbox_inches='tight', dpi=300)
    t += 1
    plt.close()
