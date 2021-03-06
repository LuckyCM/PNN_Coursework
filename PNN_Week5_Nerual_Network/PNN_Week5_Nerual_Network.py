from keras.datasets import mnist

(x_train, labels_train), (x_test, labels_test) = mnist.load_data()

############################################################
# Load Data
############################################################

# Transform the integer[0,255] to [0,1]
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Convert to one-hot encoding(for this case, 10*10 matrix)
from keras.utils import to_categorical
y_train = to_categorical(labels_train, 10)
y_test = to_categorical(labels_test, 10)

# Use data as input layer in dense-layer(full-connected), reshape the raw-data to martrix
x_train = x_train.reshape(60000, 784)   # Number of Data=60000; 28*28=784
x_test = x_test.reshape(10000, 784)     # Number of test data=10000

# Use data as convolutional layer in dense-layer(full-connected), reshape the data to 4D-martrix
# (No.exemplars, Image width, Image height, No.color-channel)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

############################################################
# Define a Neural Network
# i.e. architecture (the number and types of layers, and their interconnectivity)
# Keras allows neural networks to be defined in several different ways, two are described below.
############################################################

# Method 1:
# # MLP network(3 layers)
# from keras.models import Sequential
# from keras.layers import Dense
# net = Sequential()  # Name of MLP is net
# net.add(Dense(800, activation='relu', input_shape=(784, )))    # number of neurons=800, activation function='relu'
# net.add(Dense(400, activation='relu'))
# net.add(Dense(10, activation='softmax'))
# # Other activation function (see https://keras.io/activations/).

# CNN
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization
net = Sequential()
net.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
net.add(BatchNormalization())
net.add(Conv2D(64, (5, 5), activation='relu'))
net.add(MaxPool2D(pool_size=(3, 3)))
net.add(Flatten())
net.add(Dense(256, activation='relu'))
net.add(Dropout(rate=0.5))
net.add(Dense(10, activation='softmax'))

# Method 2:
# # MLP network(3 layers)
# from keras.models import Model
# from keras.layers import Input, Dense
# input_img = Input(shape=(784, ))     # define a placeholder for the input data
# x = Dense(800, activation='relu')(input_img)
# y = Dense(400, activation='relu')(x)
# z = Dense(10, activation='softmax')(y)
# net = Model(input_img, z)
#
# # CNN
# from keras.models import Model
# from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout, BatchNormalization, Input
# inputs = Input(shape=x_train.shape[1:])
# n = Conv2D(filters=32, kernel_size=(5, 5), activation='relu')(inputs)
# n = BatchNormalization(3)(n)
# n = Conv2D(filters=64, kernel_size=(3, 3), activation='relu')(n)
# n = MaxPool2D(pool_size=(2, 2))(n)
# n = Flatten()(n)
# n = Dense(256, activation='relu')(n)
# n = Dropout(rate=0.5)(n)
# outputs = Dense(10, activation='softmax')(n)
# net = Model(inputs, outputs)

# obtain a textual description of its structure
net.summary()


# obtain image
from keras.utils import plot_model
plot_model(net, to_file='network_structure.png', show_shapes=True)

############################################################
# Train
############################################################

# 1. Compile the network
#  cost function (https://keras.io/losses/) and the optimizer (https://keras.io/optimizers/)
net.compile(loss='categorical_crossentropy', optimizer='adam')

# 2. Fit()
history = net.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=30, batch_size=256)

# Plot cost-function change image
import matplotlib.pyplot as plt
plt.figure()
plt.plot(history.history['loss'], label='training loss')
plt.plot(history.history['val_loss'], label='validation loss')
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend()

# save the trained network model
net.save("network_for_mnist.h5")

# reload saved model
from keras.models import load_model
net = load_model("network_for_mnist.h5")

############################################################
# Test
############################################################

# Compare predict result and test result
import numpy as np
outputs = net.predict(x_test)
labels_predict = np.argmax(outputs, axis=1)
misclassified = sum(labels_predict != labels_test)
print('Percentage misclassified =', 100 * misclassified / labels_test.size)
print('Accurate =', 100 - (100 * misclassified / labels_test.size))


plt.figure(figsize=(8, 2))
for i in range(0, 8):
    ax = plt.subplot(2, 8, i+1)
    plt.imshow(x_test[i, :].reshape(28, 28), cmap=plt.get_cmap('gray_r'))
    plt.title(labels_test[i])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

for i in range(0, 8):
    # output = net.predict(x_test[i, :].reshape(1, 784)) # if MLP
    output = net.predict(x_test[i, :].reshape(1, 28, 28, 1))    # if CNN
    output = output[0, 0:]
    plt.subplot(2, 8, 8+i+1)
    plt.bar(np.arange(10.), output)
    plt.title(np.argmax(output))
    plt.show()


