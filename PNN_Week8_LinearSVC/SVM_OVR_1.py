import numpy as np
matrix_add = np.array([[89.06, -86.06, -80.1, 1],
                       [86.06, -84.5, -78.78, 1],
                       [80.1, -78.78, -73.46, 1],
                       [1, -1, -1, 0]])

matrix_y = np.array([[1],
                    [-1],
                    [-1],
                    [0]])
X = [[-9.1, 2.5], [-9.1, 1.3], [-8.5, 1.1]]
lam = np.dot(np.linalg.inv(matrix_add), (matrix_y))
y = [1, -1, -1]
W = lam[0]*y[0]*X[0] + lam[1]*y[1]*X[1] + lam[2]*y[2]*X[2]
WT = np.transpose(W)
w0 = lam[3]
# Hyperplane = (WT[0:2] * X + w0)

print("Hyperplane = ", W[0], '* x1 +', W[1], '* x2  +', w0)

import math
margin = 2 / math.sqrt( math.pow(W[0] ,2)+ math.pow(W[1] ,2))
print("Margin = 2 / ||W|| = ", margin)