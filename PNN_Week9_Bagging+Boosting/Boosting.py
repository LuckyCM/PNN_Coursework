from sklearn.datasets import make_blobs
import random
import numpy as np
from pandas import DataFrame
import matplotlib.pyplot as plt

#################################
# Question 4 - Boosting
#################################
X, y = make_blobs(n_samples=20, centers=2, random_state=9124)

# Generate M dataset D1, D2, . . ., DM.
def subsample(dataset, last_wrong_sample_index):
    sample = list()
    sample_index = list()
    subsamples_noreplacement = dataset
    ratio = random.randint(25, 80)/100
    n_sample = round(len(dataset) * ratio)
    for i in range(n_sample):
        index = random.randint(0, len(dataset)-1)  # randomly pick data
        while index in sample_index:    # no replacement
            index = random.randint( 0, len( dataset ) - 1 )  # randomly pick data

        pick_sample = subsamples_noreplacement[index]
        sample.append(pick_sample)
        sample_index.append(index)
    for n in range(len(last_wrong_sample_index)):
        if last_wrong_sample_index[n] not in sample_index:
            sample.append(dataset[n])
            sample_index.append(last_wrong_sample_index[n])
    return sample, sample_index

def Vote(dataset, y_preds, y):
    rightcount = 0
    for i in range(len(y)):

        col_pred = np.array(y_preds)[:, i]
        if col_pred[0] == col_pred[1]:
            if col_pred[0] == 0:
                flag_new = 0
            else:
                flag_new = 1
        else:
            flag_new = col_pred[2]

        if flag_new == y[i]:
            rightcount += 1
        print('Data: ', dataset[i], ', %d | Predict label is %d' % (y[i], flag_new))
    print("Accuracy is %f" % (rightcount / len(dataset)))

# Q3. Learn weak classifier for each dataset.
def train(dataset, sample, label):
    from sklearn import svm
    clf = svm.SVC(kernel='linear')
    clf.fit(sample, label)
    y_pred = clf.predict(dataset)
    return y_pred

# Q4.Combine all weak classifiers using a majority voting scheme.
from pandas import Series
from random import seed
seed(120)
data = np.array(X)
y = y
df = DataFrame(dict(x=X[:, 0], y=X[:, 1], label=y))

last_wrong_sample_index = []
for size in [3]:
    y_preds = []
    sample_votes = list()
    for i in range(size):
        sample, index = subsample(data, last_wrong_sample_index)
        label = df.iloc[:, 2][index]
        while (len(np.unique(label)) != 2):  # To make sure there are two classes
            sample, index = subsample(data, last_wrong_sample_index)
            label = df.iloc[:, 2][index]
        else:
            last_wrong_sample_index = []
            y_pred = train(data, sample, label)
            y_preds.append(y_pred)
        for n in range( len( data ) ):
            if y[n] != y_preds[i][n]:
                last_wrong_sample_index.append(n)

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