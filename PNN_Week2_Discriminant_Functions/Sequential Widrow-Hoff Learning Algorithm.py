from sklearn import datasets
import numpy as np


if __name__ == "__main__":
    # iris = datasets.load_iris()
    # # print(iris)

    n = 0.1  # learning_rate
    bT = np.mat([[1.0, 1.5, 2.5, 0.5, 1.5, 1.0]])  # margin
    b = np.transpose(bT)
    # aT = [1.0, 0.0, 0.0]
    aT = np.mat([[1.0, 0.0, 0.0]])
    a = np.transpose(aT)
    # print(a)
    # print(aT)

    y = np.mat([[1, 0.0, 2.0],
                    [1, 1.0, 2.0],
                    [1, 2.0, 1.0],
                    [-1, 3.0, -1.0],
                    [-1, 2.0, 1.0],
                    [-1, 3.0, 2.0]])  # feature_vector


    epoch = 2
    for i in range(2):
        for k in range(len(y)):
            g = aT*(np.transpose(y[k]))
            aT = aT+n*(b[k]-g)*y[k]
            print('iteration %d: %s | %s'%(k,g,aT))

