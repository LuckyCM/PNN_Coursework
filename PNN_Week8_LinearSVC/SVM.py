from sklearn import svm
import matplotlib.pyplot as plt
import mglearn
from sklearn.datasets import make_blobs
import numpy as np

###################################
# Question 3
###################################
# Generate the data points
# X include (X1, X2); y is the label of the data
X, y = make_blobs(random_state=18, n_samples=30, centers=3)

plt.figure()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel("X1")
plt.ylabel("X2")
plt.legend(["Class 0", "Class 1", "Class 2"])
plt.savefig('data_classified.png')
plt.show()
# Classifying data to 3 Classes
Class0, Class1, Class2 = [], [], []

for i in range(len(X)):
    if y[i] == 0:
        Class0.append(X[i])
    if y[i] == 1:
        Class1.append(X[i])
    if y[i] == 2:
        Class2.append(X[i])
Class0_X = np.array(Class0)
Class1_X = np.array(Class1)
Class2_X = np.array(Class2)
print("y=0:", '\n', Class0_X)
print("y=1:", '\n', Class1_X)
print("y=2:", '\n', Class2_X)

# ###################################
# # Question 4
# ###################################
def make_meshgrid(x,y,h=0.02):
    """
    x:data to base x-axis meshgrid on
    y:data to base y-axis meshgrid on
    h:stepsize for meshgrid,optional
    """
    x_min,x_max=x.min()-1,x.max()+1
    y_min,y_max=y.min()-1,y.max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,h),
                     np.arange(y_min,y_max,h))
    return xx,yy

def plot_contours(clf,xx,yy,**params):
    """
    plot the decision boundaries for a classifier
    ****************************************
    parameters:
    ax:matplotlib axes object
    clf:a classifier
    xx:meshgrid ndarray
    yy:meshgrid ndarray
    params:dictionary of params to pass to contourf,optional
    """
    Z=clf.predict(np.c_[xx.ravel(),yy.ravel()])
    Z=Z.reshape(xx.shape)
    out=plt.contourf(xx, yy, Z, **params)
    return out

C = 1.0  # SVM regularization parameter
clf = svm.LinearSVC(C=C, max_iter=10000)
# clf = svm.SVC(decision_function_shape='ovr')
model = clf.fit( X, y )

# title the plot
title = ('LinearSVC (linear kernel)')

plt.subplots_adjust( wspace=0.4, hspace=0.4 )

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid( X0, X1 )

plot_contours(clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8 )
# plt.scatter( X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k' )
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
plt.xlabel( 'Sepal length' )
plt.ylabel( 'Sepal width' )
plt.xticks( () )
plt.yticks( () )
plt.title( title )
plt.savefig('LinearSVC.png')
plt.show()

###################################
# Question 6
###################################
plt.figure()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
x = np.linspace(-100., 100.)
y1 = (22.4 - 4.20 * x) / 0.16  #  -4.20 * x1 + -0.16 * x2 + [22.4] = 0
y2 = (-2.82 + 0.35 * x) / 0.169  # 0.35 * x1 -0.169 * x2 - 2.82 = 0
y3 = (3.31-0.455*x) / 0.488  # 0.455 * x1 +0.488 * x2 - 3.31 = 0
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.xlabel( 'Sepal length' )
plt.ylabel( 'Sepal width' )
plt.xticks( () )
plt.yticks( () )
plt.title( 'One-against-all hyperplane' )
plt.savefig('One-against-all.png')
plt.show()

###################################
# Question 7
###################################
X1_new = np.array(range(10), dtype=float)
X2_new = np.array(range(10), dtype=float)
for i in range(len(Class0_X)):
    X1_new[i] = (Class0_X[i, 0]+Class1_X[i, 0]+Class2_X[i, 0])/3
    X2_new[i] = (Class0_X[i, 1]+Class1_X[i, 1]+Class2_X[i, 1])/3

plt.figure()
mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
x = np.linspace(-100., 100.)
y1 = (22.4 - 4.20 * x) / 0.16  #  -4.20 * x1 + -0.16 * x2 + [22.4] = 0
y2 = (-2.82 + 0.35 * x) / 0.169  # 0.35 * x1 -0.169 * x2 - 2.82 = 0
y3 = (3.31-0.455*x) / 0.488  # 0.455 * x1 +0.488 * x2 - 3.31 = 0
plt.plot(x, y1)
plt.plot(x, y2)
plt.plot(x, y3)
plt.scatter(X1_new, X2_new, c='red')
plt.xlabel( 'Sepal length' )
plt.ylabel( 'Sepal width' )
plt.xticks( () )
plt.yticks( () )
plt.title( 'mean-classification' )
plt.savefig('mean-classification.png')
plt.show()

print("a")
