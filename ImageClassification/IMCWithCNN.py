import sys
from PIL import Image, ImageFile
from keras.applications import VGG19, imagenet_utils, ResNet50
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras.layers import Dense, Dropout, Flatten, Activation, Conv2D
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras import backend as K
from keras.optimizers import SGD, Adam
from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import log_loss
from keras.models import Model, Sequential
import datetime

K.set_image_dim_ordering('tf')

## setting this to print to message.log file in nohup mode and check the results at a later point.
environment = 'prod'   #prod

if environment == 'prod':
    old_stdout = sys.stdout
    log_file = open("custom_cnn_message.log","w")
    sys.stdout = log_file

#################################################################
# Load the data

car_names = ['jeep', 'minivan', 'pickup', 'sports', 'truck']
new_height = 224
new_width = 224
print "start time of run", datetime.datetime.now() # noting when the run begins

# load images and target names
def load_dataset(path):
    data = load_files(path, categories=car_names)
    vehicle_files = np.array(data['filenames'])

    vehicle_targets = np.array(data['target'])
    le = LabelEncoder()
    vehicle_targets = le.fit_transform(vehicle_targets)
    vehicle_targets = np_utils.to_categorical(vehicle_targets)
    return vehicle_files, vehicle_targets

train_files, train_targets = load_dataset('../data/train')
valid_files, valid_targets = load_dataset('../data/valid')
test_files, test_targets = load_dataset('../data/test')

ImageFile.LOAD_TRUNCATED_IMAGES = True

def load_image(infilename):
    data = image.load_img(infilename, target_size=(224, 224))
    data = image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)
    return data

#### train array
train_array = np.empty([len(train_files), new_height, new_width, 3])
i = 0
for x in train_array:
    train_array[i] = load_image(train_files[i])
    i += 1
train_array=np.array(train_array)

#### test array
test_array = np.empty([len(test_files), new_height, new_width, 3])
i = 0
for x in test_array:
    test_array[i] = load_image(test_files[i])
    i += 1
test_array=np.array(test_array)

#### valid array
valid_array = np.empty([len(valid_files), new_height, new_width, 3])
i = 0
for x in valid_array:
    valid_array[i] = load_image(valid_files[i])
    i += 1
valid_array=np.array(valid_array)

####### Set X and y train and test/valid
X_train = train_array
X_train = X_train.astype('float32')
X_train /= 255
y_train = np.array(train_targets)
print "X_train", X_train.shape
print "y_train", y_train.shape

X_test = test_array
X_test = X_test.astype('float32')
X_test /= 255
y_test = np.array(test_targets)
print "X_test", X_test.shape
print "y_test", y_test.shape

X_valid = valid_array
y_valid = np.array(valid_targets)
print "X_valid", X_valid.shape
print "y_valid", y_valid.shape

# Create the model
def create_model(img_rows, img_cols, channel=1, num_classes=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(new_height, new_width, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 5
batch_size = 20
nb_epoch = 50

# Load our model
model = create_model(img_rows, img_cols, channel, num_classes)
print "Model Summary", model.summary()

# # Start training
model.fit(X_train, y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_test, y_test))

scores = model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))

model_json = model.to_json()
with open("custom_model.json", "w") as json_file:
     json_file.write(model_json)
model.save_weights("custom_weights.h5")
print("Saved model to disk")

# Make predictions
predictions_test = model.predict(X_test, batch_size=batch_size, verbose=1)
print "prediction"

# Categorical cross entropy loss score
score = log_loss(y_test, predictions_test)

if environment == 'prod':
    sys.stdout = old_stdout
    log_file.close()
