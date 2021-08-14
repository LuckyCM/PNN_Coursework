import numpy as np
matrix_add = np.array([[-26.05, 44.35, -32.61, 1],
                       [-44.35, 97.25, -41.75, 1],
                       [-32.61, 41.75, -49.54, 1],
                       [-1, 1, -1, 0]])

matrix_y = np.array([[-1],
                    [1],
                    [-1],
                    [0]])
X = [[5.1, -0.2], [8.5, -5.0], [6.5, 2.7]]
lam = np.dot(np.linalg.inv(matrix_add), (matrix_y))
y = [-1, 1, -1]
W = lam[0]*y[0]*X[0] + lam[1]*y[1]*X[1] + lam[2]*y[2]*X[2]
WT = np.transpose(W)
w0 = lam[3]
# Hyperplane = (WT[0:2] * X + w0)

print("Hyperplane = ", W[0], '* x1 +', W[1], '* x2  +', w0)

import math
margin = 2 / math.sqrt( math.pow(W[0] ,2)+ math.pow(W[1] ,2))
print("Margin = 2 / ||W|| = ", margin)