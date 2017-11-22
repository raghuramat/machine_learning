import argparse
import sys
from PIL import Image, ImageFile, ImageDraw
from keras.applications import ResNet50
from keras.preprocessing import image as image_utils
from keras.applications.imagenet_utils import decode_predictions
from sklearn.datasets import load_files
from keras.utils import np_utils
import numpy as np
from keras import backend as K
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from sklearn.preprocessing import LabelEncoder
from keras.models import Model, model_from_json
import datetime

K.set_image_dim_ordering('th')

## setting this to print to message.log file in nohup mode and check the results at a later point.
environment = 'test'   #prod

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

# construct the argument parse and parse the arguments
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True,
#                 help="path to the input image")
# args = vars(ap.parse_args())

image_path = 'data/test/minivan/minivan-4352.jpg'
image_type = 'jeep'

# load the original image via OpenCV so we can draw on it and display
# it to our screen later
# print "image to load", args["image"]
orig = Image.open(image_path)
# orig = Image.open(args["image"])

def load_image(infilename):
    data = image.load_img(infilename, target_size=(224, 224))
    data = image.img_to_array(data)
    data = np.expand_dims(data, axis=0)
    data = preprocess_input(data)
    return data

# Create the model
def create_model(img_rows, img_cols, channel=1, num_classes=None):
    model = ResNet50(weights='imagenet', include_top=True)
    # model.load_weights('weights/vgg16_weights_th_dim_ordering_th_kernels.h5')  # Loading weights of vgg16.
    # trying to fine tune by popping last layer and adding a softmax
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

img_rows, img_cols = 224, 224 # Resolution of inputs
channel = 3
num_classes = 5
batch_size = 20
nb_epoch = 30

# Load our model
model = create_model(img_rows, img_cols, channel, num_classes)
#print "Model Summary", model.summary()

print("[INFO] loading and preprocessing image...")
image = load_image(image_path)
print "shape of image", image.shape
print("[INFO] classifying image...")
preds = model.predict(image)
print "Prediction shape", preds.shape
print preds
#
# (inID, label) = decode_predictions(preds)[0]
#
# print("ImageNet ID: {}, Label: {}".format(inID, label))
# orig.text((10,30), "Label: {}".format(label), fill='green')

# orig.show()
print orig.show()
# print "Image Shown"
# cv2.putText(orig, "Label: {}".format(label), (10, 30),
#     cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
# cv2.imshow("Classification", orig)
# cv2.waitKey(0)

if environment == 'prod':
    sys.stdout = old_stdout
    log_file.close()
