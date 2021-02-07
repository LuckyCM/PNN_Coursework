import numpy as np


if __name__ == "__main__":
    vector_mat = np.mat([[1, 0.0, 2.0],
                        [1, 1.0, 2.0],
                        [1, 2.0, 1.0],
                        [0, -3.0, 1.0],
                        [0, -2.0, -1.0],
                        [0, -3.0, -2.0]])
    x = vector_mat[:, 1:]
    xT = np.transpose(x)
    c1 = np.mat(np.ones(6))
    x_augmented = np.column_stack((np.transpose(c1), x))
    tT = np.transpose(vector_mat[:, 0])
    t = np.transpose(tT)

    n = 1.0  # learning_rate
    w = np.mat([[5.5, -7.5, -0.5]])
    epoch = 2
    theta = -5.5
    for i in range(epoch):
        for k in range(len(x)):
            criterion = w*np.transpose( x_augmented[k] )
            if criterion > 0:
                result = 1
            elif criterion < 0:
                result = 0
            elif criterion == 0:
                result = 0.5
            w = w + n * (t[k] -result)*x_augmented[k]
            print( 'iteration %d: %s | %s' % (k,result,w) )



