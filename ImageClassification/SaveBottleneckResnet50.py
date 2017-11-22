import numpy as np
import sys
from PIL import ImageFile
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications

## setting this to print to message.log file in nohup mode and check the results at a later point.
environment = 'prod'   #prod

if environment == 'prod':
    old_stdout = sys.stdout
    log_file = open("Resnet50_message.log","w")
    sys.stdout = log_file

# dimensions of our images.
img_width, img_height = 224, 224

top_model_weights_path = 'resnet50_bottleneck_fc_model.h5'
train_data_dir = '../data/train'
validation_data_dir = '../data/valid'
nb_train_samples = 2736
nb_validation_samples = 592
epochs = 50
batch_size = 16
ImageFile.LOAD_TRUNCATED_IMAGES = True

def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    np.save(open('resnet50_bottleneck_features_train.npy', 'w'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    np.save(open('resnet50_bottleneck_features_validation.npy', 'w'),
            bottleneck_features_validation)


def train_top_model():
    train_data = np.load(open('resnet50_bottleneck_features_train.npy'))
    train_labels = np.array(
        [0] * (nb_train_samples / 2) + [1] * (nb_train_samples / 2))

    validation_data = np.load(open('resnet50_bottleneck_features_validation.npy'))
    validation_labels = np.array(
        [0] * (nb_validation_samples / 2) + [1] * (nb_validation_samples / 2))

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, train_labels,
              epochs=epochs,
              batch_size=batch_size,
              validation_data=(validation_data, validation_labels))

    model_json = model.to_json()
    with open("resnet50_bottleneck_features_train.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights(top_model_weights_path)


save_bottlebeck_features()
train_top_model()

if environment == 'prod':
    sys.stdout = old_stdout
    log_file.close()