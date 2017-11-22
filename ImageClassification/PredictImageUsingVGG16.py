import argparse
import sys
from PIL import Image, ImageFile, ImageDraw
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.models import Model, model_from_json
import datetime
from keras.optimizers import SGD, Adam
K.set_image_dim_ordering('th')

environment = 'test'   #prod
image_type = 'minivan'
image_filename = 'minivan-4406'
image_path = 'data/test/' + image_type + '/' + image_filename + '.jpg'
cnn_type = 'vgg16'

if environment == 'prod':
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True,
                    help="path to the input image")
    ap.add_argument("-t", "--type", required=True,
                    help="type of image")
    args = vars(ap.parse_args())
    image_filename = args["image"]
    image_type = args["type"]
    image_path = 'data/test/' + image_type + '/' + image_filename + '.jpg'

old_stdout = sys.stdout
log_path = 'results/' + image_filename + '_' + cnn_type + '.log'
log_file = open(log_path, "w")
sys.stdout = log_file

#################################################################
# Load the data

car_names = ['jeep', 'minivan', 'pickup', 'sports', 'truck']
new_height = 224
new_width = 224
print "start time of run", datetime.datetime.now() # noting when the run begins

# load the original image via OpenCV so we can draw on it and display
# it to our screen later
orig = Image.open(image_path)

def load_image(infilename):
    data = image.load_img(infilename, target_size=(224, 224))
    data = image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)
    return data

# Create the model
def create_model():
    json_file = open('model_created/VGG16_Modified.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model_created/VGG16_Modified.h5")
    print("Loaded model from disk")

    # evaluate loaded model on test data
    sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
    loaded_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    return loaded_model


# Load our model
model = create_model()
image = load_image(image_path)

print("[INFO] classifying image...")
preds = model.predict(image)
print "Prediction", preds[0]

index = 0
for x in range(0,4):
    if(car_names[x] == image_type):
        index = x

print("%s: %.2f%%" % (image_type, preds[0][index]*100))

if environment == 'test':
    # orig.text((10,30), "Label: {}".format(image_type), fill='green')
    orig.show()

if environment == 'prod':
    sys.stdout = old_stdout
    log_file.close()
