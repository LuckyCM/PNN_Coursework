import numpy as np

X = [[5.1, -0.2], [8.5, -5.0], [6.5, 2.7]]
y = [-1, 1, -1]

X = np.array(X)
y = np.array(y)


# W = lam[1]*y[1]*X[1] + lam[2]*y[2]*X[2] + lam[3]*y[3]*X[3]
W = [[], [], []]
WT = np.transpose(W)
XT = np.transpose(X)

# Substitute into the equation: yi * (WTX + w0) = 1
# x = x1, y1 = 1:

matrix0 = X * X[0]
matrix_add1 = matrix0[:, 0]+matrix0[:, 1]
matrix_add1 = matrix_add1 * y
matrix1 = X * X[1]
matrix_add2 = matrix1[:, 0]+matrix1[:, 1]
matrix_add2 = matrix_add2 * y
matrix2 = X * X[2]
matrix_add3 = matrix2[:, 0]+matrix2[:, 1]
matrix_add3 = matrix_add3 * y

matrix_add = np.array([[matrix_add1, 1],
                      [matrix_add2, 1],
                      [matrix_add3, 1]])


print("success")
print(matrix_add)
