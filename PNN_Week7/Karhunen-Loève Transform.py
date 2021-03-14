from sklearn import datasets
import numpy as np

#######################################
# Load data
#######################################
iris = datasets.load_iris()
X = iris.data

#######################################
# PCA
#######################################
from sklearn.decomposition import PCA

pca = PCA(n_components=2)

pca.fit(X)

X_new = np.array([[5.3, 3.9, 2.0, 0.2],
         [7.1, 4.2, 3.9, 0.5],
         [6.0, 2.8, 3.2, 1.9],
         [5.8, 4.1, 4.9, 1.2],
         [7.2, 3.6, 5.1, 0.8]])
result = pca.transform(X_new)

print(result)