import random
import matplotlib.pyplot as plt
from pandas import DataFrame
import numpy as np
from sklearn.datasets import make_blobs

#################################
# Question 3 - Bagging
#################################
# Q1. Start with dataset D.
X, y = make_blobs(n_samples=20, centers=2, random_state=9124)

# Q2. Generate M dataset D1, D2, . . ., DM.
def subsample(dataset):
    sample = list()
    sample_index = list()
    ratio = random.randint(25, 100)/100
    n_sample = round(len(dataset) * ratio)
    for i in range(n_sample):
        index = random.randint(0, len(dataset)-1)  # randomly pick data
        sample.append(dataset[index])
        sample_index.append(index)
    return sample, sample_index

# You can use voting or average at here
# def mean(numbers):
#     return sum(numbers)/float(len(numbers))
def Vote(dataset, y_preds, y):
    rightcount = 0
    for i in range(len(y)):

        col_pred = np.array(y_preds)[:, i]
        n0 = np.sum(col_pred == 0)
        n1 = np.sum(col_pred == 1)
        if n0 > n1:
            flag_new = 0
        else:
            flag_new = 1
        if flag_new == y[i]:
            rightcount += 1
        print('Data: ',dataset[i],', %d | Predict label is %d' % (y[i], flag_new))
    print("Accuracy is %f" % (rightcount / len(dataset)))

# Q3. Learn weak classifier for each dataset.
def trainAndPlot(dataset, sample, label, size):
    from sklearn import svm
    clf = svm.SVC(kernel='linear')
    clf.fit(sample, label)
    y_pred = clf.predict(dataset)

    w = clf.coef_[0]
    a = -w[0]/w[1] # k

    # Plot the line
    xx = np.linspace(-11, -5)
    yy = a * xx - (clf.intercept_[0])/w[1]
    print('Hyperplane is: y = %f * x - %f'% (a, (clf.intercept_[0])/w[1]))
    df = DataFrame( dict( x=np.transpose(sample)[0], y=np.transpose(sample)[1], label=label ) )
    colors = {0: 'red', 1: 'green'}
    markers = {0: 'x', 1: 'o'}
    fig, ax = plt.subplots()
    grouped = df.groupby( 'label' )
    for key, group in grouped:
        group.plot( ax=ax, kind='scatter', x='x', y='y', label=key, color=colors[key], marker=markers[key] )

    # plt.scatter( clf.support_vectors_[:, 0], clf.support_vectors_[:, 1], s=80 )
    # plt.scatter( [x[0] for x in sample], [x[1] for x in sample], c=[colors[i] for i in label],marker=[markers[i] for i in label])
    plt.plot(xx, yy)
    plt.xlim( -11, -5 )
    plt.ylim( -2, 4.5 )
    plt.title('size='+str(size)+ ', n_sample='+ str(len(sample)))
    plt.axis('tight')
    plt.legend( ['Hyperplane', "Class -1", "Class +1"] )
    plt.savefig('./figure/'+str(size)+'_Classifiers/'+str(len(sample))+'_samples.png')
    plt.show()

    return y_pred


# Q4.Combine all weak classifiers using a majority voting scheme.
from pandas import Series
from random import seed
seed(120)
data = np.array(X)
y = y
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))
for size in [3, 4, 5]:
    y_preds = []
    sample_votes = list()
    for i in range(size):
        sample, index = subsample(data)
        label = df.iloc[:, 2][index]
        while (len(np.unique(label)) != 2):  # To make sure there are two classes
            sample, index = subsample(data)
            label = df.iloc[:, 2][index]
        else:
            y_pred = trainAndPlot(data, sample, label, size)
            y_preds.append(y_pred)
    # Weak classifier
    print( "----------------------------------------" )
    for index in range( size ):
        rightcount = 0
        print( 'For ', size, ' weak classifer', ' No.', index + 1, ':' )
        for i in range(len(data)):
            if y[i] == y_preds[index][i]:
                rightcount += 1
            print('Data: ', data[i], ', %d | Predict label is %d' % (y[i], y_preds[index][i]) )
        print( "Accuracy is %f" % (rightcount / len( data )) )

    # Overall classifier
    print('Overall Classifer ' + str( size ) + ':')
    sample_vote = Vote(data, y_preds, y)
    sample_votes.append(sample_vote)
    print( "****************************************" )