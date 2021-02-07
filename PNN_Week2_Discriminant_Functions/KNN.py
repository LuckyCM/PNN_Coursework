from sklearn import neighbors
from sklearn import datasets
import numpy as np

if __name__ == "__main__":

    data = np.mat( [[6.2, 3.0, 1.1, 1.4],
                    [5.5, 2.2, 1.9, 1.0],
                    [4.5, 2.7, 4.2, 1.6],
                    [5.1, 4.4, 1.6, 0.9],
                    [5.4, 2.9, 5.1, 1.5]])

    knn = neighbors.KNeighborsClassifier(5)

    iris = datasets.load_iris()
    knn.fit(iris.data,iris.target)
    for n in range( len( data ) ):
        predictlabel = knn.predict(data[n])
        print(predictlabel)