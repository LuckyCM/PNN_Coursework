from sklearn.decomposition import sparse_encode
import numpy as np
from sklearn import datasets
from sklearn import neighbors

numNonZero = 2
tolerance = 1.000000e-05

knn = neighbors.KNeighborsClassifier(5)

iris = datasets.load_iris()
knn1 = knn.fit(iris.data, iris.target)


V1 = iris.data[0:50]
V2 = iris.data[50:100]
V3 = iris.data[100:150]
Vt = np.transpose(V2)

x = np.mat([[7.5, 2.2, 6.3, 1.8],
            [7.2, 2.5, 3.2, 0.4],
            [5.0, 4.3, 6.4, 2.1],
            [4.7, 4.3, 1.2, 0.7],
            [6.4, 4.1, 2.1, 1.5]])

λ = 0.1
cost = []
xt = np.transpose(x)
for i in range(5):
    print(x[i])
    y = sparse_encode(x[i], V2, algorithm='omp', n_nonzero_coefs=numNonZero, alpha=tolerance)
    yt = np.transpose(y)

    # cost = np.sqrt(((x - (Vt * y))[0])**2 + ((x - (Vt * y))[1])**2) + λ * np.count_nonzero(y)
    cost.append(np.linalg.norm((x[i] - np.transpose(np.dot(Vt, yt))), ord=2, axis=1) + λ * np.count_nonzero(y))

print(str(cost))
###########################################
# sample class
###########################################
for n in range( len( x ) ):
    predictlabel = knn.predict( x[n] )
    print( predictlabel )