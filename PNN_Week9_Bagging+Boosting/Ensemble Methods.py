from random import randrange

from sklearn.datasets._samples_generator import make_blobs
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from collections import Counter
#################################
# Question 1
#################################
# generate 2d classification dataset
X, y = make_blobs(n_samples=20, centers=2, random_state=9124)
# Classifying data to 2 Classes
Class0, Class1 = [], []

for i in range(len(X)):
    if y[i] == 0:
        Class0.append(X[i])
    if y[i] == 1:
        Class1.append(X[i])
Class0_X = np.array(Class0)
Class1_X = np.array(Class1)
print("y=0:", '\n', Class0_X)
print("y=1:", '\n', Class1_X)

#################################
# Question 2
#################################
# scatter plot, dots colored by class value
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
colors = {0:'red', 1:'green'}
markers = {0:'x', 1:'o'}

fig, ax = plt.subplots()
grouped = df.groupby('label')
for key, group in grouped:
      group.plot(ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key], marker=markers[key])
plt.xlabel("X1")
plt.ylabel("X2")
plt.xlim(-11, -5)
plt.ylim(-2, 4.5)
plt.title("non-linear separable dataset")
plt.legend(["Class -1", "Class +1"])
plt.savefig('./figure/data_classified.png')
plt.show()



