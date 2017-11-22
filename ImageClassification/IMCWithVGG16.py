import sys
from PIL import Image, ImageFile
from keras.applications import VGG19, imagenet_utils
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
import pandas as pd
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

K.set_image_dim_ordering('th')

## setting this to print to message.log file in nohup mode and check the results at a later point.
environment = 'prod'   #prod

if environment == 'prod':
    old_stdout = sys.stdout
    log_file = open("message.log","w")
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
train_array = np.empty([len(train_files), 3, new_height, new_width])
i = 0
for x in train_array:
    train_array[i] = load_image(train_files[i])
    i += 1
train_array=np.array(train_array)

#### test array
test_array = np.empty([len(test_files), 3, new_height, new_width])
i = 0
for x in test_array:
    test_array[i] = load_image(test_files[i])
    i += 1
test_array=np.array(test_array)

#### valid array
valid_array = np.empty([len(valid_files), 3, new_height, new_width])
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
    model = VGG16(weights='imagenet', include_top=False)

    custom_model = Sequential()
    custom_model.add(Conv2D(32, (3, 3), input_shape=model.output_shape[1:]))
    custom_model.add(Activation('relu'))
    custom_model.add(MaxPooling2D(pool_size=(2, 2)))

    custom_model.add(Conv2D(32, (3, 3)))
    custom_model.add(Activation('relu'))
    custom_model.add(MaxPooling2D(pool_size=(2, 2)))

    custom_model.add(Conv2D(64, (3, 3)))
    custom_model.add(Activation('relu'))
    custom_model.add(MaxPooling2D(pool_size=(2, 2)))

    custom_model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    custom_model.add(Dense(64))
    custom_model.add(Activation('relu'))
    custom_model.add(Dropout(0.5))
    custom_model.add(Dense(num_classes))
    custom_model.add(Activation('sigmoid'))

    custom_model.load_weights('model_created/custom_weights.h5')
    model = Model(input=model.input, output=custom_model(model.output))

    for layer in model.layers[:25]:
        layer.trainable = False

    # Using sgd works at batch size 16. Using adam even at batch size 8 is taking the instance OOM
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

    return model

img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 5
batch_size = 20
nb_epoch = 45

# Load our model
model = create_model(img_rows, img_cols, channel, num_classes)
print "Model Summary", model.summary()

# # Start training
model.fit(X_train, y_train,batch_size=batch_size,epochs=nb_epoch,shuffle=True,verbose=1,validation_data=(X_valid, y_valid))

model_json = model.to_json()
with open("VGG16_Modified.json", "w") as json_file:
     json_file.write(model_json)
model.save_weights("VGG16_Modified.h5")
print("Saved model to disk")

# Make predictions
predictions_test = model.predict(X_test, batch_size=batch_size, verbose=1)
print "prediction"

P = imagenet_utils.decode_predictions(predictions_test)
for (i, (imagenetID, label, prob)) in enumerate(P[0]):
	print("{}. {}: {:.2f}%".format(i + 1, label, prob * 100))

# Categorical cross entropy loss score
score = log_loss(y_test, predictions_test)
print("Test set accuracy for %0.2f lrate and %d epoch: %.2f%%" % (1e-3, nb_epoch, (score[1] * 100)))

if environment == 'prod':
    sys.stdout = old_stdout
    log_file.close()
