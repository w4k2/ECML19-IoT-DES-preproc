# Libraries
import matplotlib.pyplot as plt
import pandas as pd
from math import pi
import numpy as np

# Set data
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

# ------- PART 1: Create background

# number of variable
df = dfincg
categories = list(df)[1:]
N = len(categories)

# What will be the angle of each axis in the plot? (we divide the plot / number of variable)
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Initialise the spider plot
ax = plt.subplot(111, polar=True)

# If you want the first axis to be on top:
ax.set_theta_offset(pi / 2)
ax.set_theta_direction(-1)

# Draw one axe per variable + add labels labels yet
plt.xticks(angles[:-1], categories)

for label in ax.get_xticklabels():
    label.set_rotation(120)

# Draw ylabels
ax.set_rlabel_position(0)
plt.yticks(
        [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.8, 0.85, 0.9],
        ["50%", "55%", "60%", "65%", "70%", "75%", "80%", "85%", "90%"],
        fontsize=6,
    )
plt.ylim(0.5, 0.90)



# ------- PART 2: Add plots

# Plot each individual = each line of the data
# I don't do a loop, because plotting more than 3 groups makes the chart unreadable

# Ind1
values = df.loc[0].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="None")
# ax.fill(angles, values, 'b', alpha=0.1)

# Ind2
values = df.loc[1].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="SMOTE")
# ax.fill(angles, values, 'r', alpha=0.1)

# Ind2
values = df.loc[2].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="SVM-SMOTE")
# ax.fill(angles, values, 'g', alpha=0.1)

# Ind2
values = df.loc[3].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="B1-SMOTE")
# ax.fill(angles, values, 'g', alpha=0.1)

# Ind2
values = df.loc[4].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="B2-SMOTE")
# ax.fill(angles, values, 'g', alpha=0.1)

# Ind2
values = df.loc[5].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="SL-SMOTE")
# ax.fill(angles, values, 'g', alpha=0.1)

# Ind2
values = df.loc[6].drop('group').values.flatten().tolist()
values += values[:1]
ax.plot(angles, values, linewidth=1, linestyle='solid', label="ADASYN")
# ax.fill(angles, values, 'g', alpha=0.1)

# Add legend
plt.legend(loc="lower center", ncol=4, columnspacing=1, frameon=False, bbox_to_anchor=(0.5, -0.2))
# Add a title
plt.title("incremental drift - G-mean", size=11, y=1.08)


plt.savefig("radarincg.eps", bbox_inches='tight')
