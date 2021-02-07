from sklearn import datasets
import numpy as np


if __name__ == "__main__":
    iris = datasets.load_iris()
    xT = iris.data
    x = np.transpose(xT)
    c1 = np.mat(np.ones(50))
    c2 = np.mat(np.zeros(100))
    c = np.column_stack((c1, c2))

    vector_mat = np.column_stack((np.transpose(c), xT))
    vector_mat = np.mat(vector_mat)
    vector_matT = np.transpose(vector_mat)

    x = vector_mat[:, 1:]
    # xT = np.transpose( x )
    c1 = np.mat( np.ones(150) )
    x_augmented = np.column_stack( (np.transpose( c1 ), x) )
    tT = np.transpose( vector_mat[:, 0] )
    t = np.transpose( tT )

    n = 0.1  # learning_rate
    w = np.mat( [[0.5, 0.5, 1.5, -1.5, 0.5]] )
    epoch = 2
    theta = -0.5
    correct = 0

    for i in range(2):
        for k in range( len( x ) ):
            criterion = w * np.transpose( x_augmented[k] )
            if criterion > 0:
                result = 1
            elif criterion < 0:
                result = 0
            elif criterion == 0:
                result = 0.5
            if result == t[k]:
                correct = correct + 1
    print( correct / 300 )

    for i in range( epoch ):
        for k in range( len( x ) ):
            criterion = w * np.transpose( x_augmented[k] )
            if criterion > 0:
                result = 1
            elif criterion < 0:
                result = 0
            elif criterion == 0:
                result = 0.5
            w = w + n * (t[k] - result) * x_augmented[k]
            print( 'iteration %d: %s | %s' % (k, result, w) )

    correct = 0
    for i in range(2):
        for k in range( len( x ) ):
            criterion = w * np.transpose( x_augmented[k] )
            if criterion > 0:
                result = 1
            elif criterion < 0:
                result = 0
            elif criterion == 0:
                result = 0.5
            if result == t[k]:
                correct = correct + 1
    print( correct / 300 )