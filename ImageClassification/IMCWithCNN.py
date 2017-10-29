import os
from PIL import Image, ImageFile
import numpy as np
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from glob import glob
from sklearn import cross_validation
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
# import matplotlib.pyplot as plt
# %matplotlib inline
# import cv2
from sklearn.grid_search import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint as sp_randint
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras import backend as K

K.set_image_dim_ordering('th')

#################################################################
# Load the data

car_names = ['jeep', 'minivan', 'pickup', 'sports', 'truck']


def load_dataset(path):
    data = load_files(path, categories=car_names)
    vehicle_files = np.array(data['filenames'])
    vehicle_targets = np_utils.to_categorical(np.array(data['target']),
                                              len(car_names))
    return vehicle_files, vehicle_targets


train_files, train_targets = load_dataset('data/train')
valid_files, valid_targets = load_dataset('data/valid')
test_files, test_targets = load_dataset('data/test')

print('There are %d total car categories.' % len(car_names))
print('There are %s total car images.\n' % len(np.hstack([train_files, valid_files, test_files])))
print('There are %d training car images.' % len(train_files))
print('There are %d validation car images.' % len(valid_files))
print('There are %d test car images.' % len(test_files))

print train_files[0]

i = 0
width = np.empty(len(train_files))
height = np.empty(len(train_files))

while i < len(train_files):
    im = Image.open(train_files[i])
    width[i], height[i] = im.size
    i += 1

new_width = int(np.amin(width))
new_height = int(np.amin(height))

print new_width
print new_height

ImageFile.LOAD_TRUNCATED_IMAGES = True


def resize_image(infilename):
    im = Image.open(infilename)
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    im.save(infilename)
    return infilename


def load_image(infilename):
    img = Image.open(infilename)
    img.load()
    data = np.asarray(img, dtype="int32")
    return data


######## resize training files and load into array
i = 0
##for img in train_files:
##    train_files[i] = resize_image(img)
##    i+=1
##    print i
print "Train images resized"
train_array = np.empty([len(train_files), new_height, new_width, 3])
# train_array = np.empty([2751,113,182,3])

i = 0
for x in train_array:
    train_array[i] = load_image(train_files[i])
    i += 1

####### resize test files and load into array
i = 0
for img in test_files:
    test_files[i] = resize_image(img)
    i += 1
print "Test images resized"
# print test_files[15]

test_array = np.empty([len(test_files), new_height, new_width, 3])

i = 0
for x in test_array:
    test_array[i] = load_image(test_files[i])
    i += 1

# print test_array[15]

######## resize validation files and load into array
i = 0
for img in valid_files:
    valid_files[i] = resize_image(img)
    i += 1

print "Validation images resized"
valid_array = np.empty([len(valid_files), new_height, new_width, 3])

i = 0
for x in valid_array:
    valid_array[i] = load_image(valid_files[i])
    i += 1

####### Set X and y train and test/valid
X_train = train_array
y_train = train_targets
print "X_train", X_train.shape
print "y_train", y_train.shape

X_test = test_array
y_test = test_targets
print "X_test", X_test.shape
print "y_test", y_test.shape

X_valid = valid_array
y_valid = valid_targets
print "X_valid", X_valid.shape
print "y_valid", y_valid.shape

################################################################################################3

print np.amin(X_train[0])
print np.amax(X_train[0])
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
# normalize inputs
X_train = X_train / 255
X_test = X_test / 255

print X_train.shape
print X_train.shape[0]
print np.amin(X_train[0])
print np.amax(X_train[0])

num_classes = y_test.shape[1]

# Create the model
model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(113, 182, 3), padding='same',
                 activation='relu'))  # , kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))  # , kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu'))  # , kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))
# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

for lrate in [0.005, 0.01, 0.015, 0.02]:
    for epoch in [40, 50, 60]:
        # epochs = 60
        # lrate = 0.02
        # decay = lrate/epochs
        # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
        # model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epoch, batch_size=32)
        # Evaluation of the model on validation set
        scores = model.evaluate(X_valid, y_valid, verbose=0)
        print("Test set accuracy for %0.2f lrate and %d epoch: %.2f%%" % (lrate, epoch, (scores[1] * 100)))
        # Evaluation of the model on test set
        scores = model.evaluate(X_test, y_test, verbose=0)
        print("Test set accuracy for %0.2f lrate and %d epoch: %.2f%%" % (lrate, epoch, (scores[1] * 100)))