from sklearn import datasets
import numpy as np


if __name__ == "__main__":
    iris = datasets.load_iris()
    xT = iris.data
    x = np.transpose(xT)
    c1 = np.mat(np.ones(50))
    c2 = np.mat(np.negative(np.ones(100)))
    c = np.column_stack((c1, c2))

    y = np.column_stack((np.transpose(c), xT))
    y = np.mat(y)
    y[50:, 1:] = np.negative(y[50:, 1:])  # normalisation
    yT = np.transpose(y)

    n = 0.01  # learning_rate
    bT = np.ones(150)  # margin
    b = np.transpose(bT)

    w0 = 0.5
    w = np.mat([[-1.5, 1.5, 1.5, -1.5]])
    aT = np.append(w0, w)   # a = [w0 wt]t
    aT = np.mat(aT)
    a = np.transpose(aT)
    # print(a)
    # print(aT)
    correct_num_pre = 0
    correct_num_after = 0
    epoch = 2
    for i in range( epoch ):
        for k in range( len( y ) ):
            g = aT * (np.transpose( y[k] ))
            if g > 0:
                correct_num_pre = correct_num_pre+1;
    print(correct_num_pre/300)



    for i in range(epoch):
        for k in range( len( y ) ):
            g = aT * (np.transpose( y[k] ))
            aT = aT + n * (b[k] - g) * y[k]
            print( 'iteration %d: %s | %s' % (k, g, aT) )


    for i in range(epoch):
        for k in range( len( y ) ):
            g = aT * (np.transpose( y[k] ))
            if g > 0:
                correct_num_after = correct_num_after+1;
    print(correct_num_after/300)
