from __future__ import absolute_import
from __future__ import print_function
import os
import itertools
import numpy as np
np.random.seed(1337)  # for reproducibility

from keras.datasets import cifar100
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.layers import BatchNormalization
from keras.optimizers import SGD, Adadelta, Adagrad
import keras.backend as K
from keras.constraints import maxnorm
K.set_image_dim_ordering('th')

'''
    Train a convnet on the CIFAR100 dataset.
    Like CIFAR10, CIFAR100 is also a labeled subset of the
    ``80 million tiny images'' dataset. But this version
    has 100 target classes each of which is in one of
    20 superclasses.
    http://www.cs.toronto.edu/~kriz/cifar.html    
    Antonio Torralba, Rob Fergus and William T. Freeman,
       *80 million tiny images: a large dataset for non-parametric
       object and scene recognition*, Pattern Analysis and Machine
       Intelligence, IEEE Transactions on 30.11 (2008): 1958-1970.
    Alex Krizhevsky, *Learning Multiple Layers of Features
       from Tiny Images*, technical report, 2009.
    This could be an interesting use of Keras graph capabilities
    with data going to two different softmax classifiers. Instead
    here we run twice and use transfer learning from the first model.
    This version can get to 53.14% test accuracy after 12 epochs.
    The final_labels version then gets 44.02% test accuracy (with
    100 classses!) following on with another 12 epochs.
    23 seconds per epoch on a GeForce GTX 680 GPU.
'''

batch_size = 128
# building model with too many classes inially for simpler transfer learning afterwards
nb_classes = 100
nb_epoch = 50

(X_train, y_train), (X_test, y_test) = cifar100.load_data(label_mode='fine')

# print shape of data while model is building
print("{1} train samples, {2} channel{0}, {3}x{4}".format("" if X_train.shape[1] == 1 else "s", *X_train.shape))
print("{1}  test samples, {2} channel{0}, {3}x{4}".format("" if X_test.shape[1] == 1 else "s", *X_test.shape))

# input image dimensions
_, img_channels, img_rows, img_cols = X_train.shape

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)
model = Sequential()
model.add(Conv2D(input_shape=(img_channels,img_rows,img_cols), filters=96, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(filters=96, kernel_size=(3,3), strides=2))
model.add(Activation('relu'))
model.add(Dropout(0.2))
model.add(Conv2D(filters=192, kernel_size=(3,3)))
model.add(Activation('relu'))
model.add(Conv2D(filters=192, kernel_size=(3,3), strides=2))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(nb_classes, activation="softmax"))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# let's train the model using SGD + momentum (how original).
sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

# a hackish transfer learning scenario - now use different labels
(z_train,), (z_test,) = cifar10.load_data()
# convert class vectors to binary class matrices
Z_train = np_utils.to_categorical(z_train, nb_classes)
Z_test = np_utils.to_categorical(z_test, nb_classes)

model.fit(X_train, Z_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Z_test))
score = model.evaluate(X_test, Z_test, show_accuracy=True, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
model.save('cifar100.h5')
