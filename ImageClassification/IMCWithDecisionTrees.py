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

#################################################################
# Load the data

car_names = ['jeep', 'minivan', 'pickup', 'sports', 'truck']


def load_dataset(path):
    data = load_files(path, categories=car_names)
    vehicle_files = np.array(data['filenames'])
    vehicle_targets = np_utils.to_categorical(np.array(data['target']),
                                              len(car_names))
    return vehicle_files, vehicle_targets


train_files, train_targets = load_dataset('../../data/asha/train')
valid_files, valid_targets = load_dataset('../../data/asha/valid')
test_files, test_targets = load_dataset('../../data/asha/test')

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

# print test_files[15]

test_array = np.empty([len(test_files), new_height, new_width, 3])

i = 0
for x in test_array:
    test_array[i] = load_image(test_files[i])
    i += 1

# print test_array[15]

####### Set X and y train and test
X_train = train_array.reshape(len(train_files), -1)
y_train = train_targets
print "X_train", X_train.shape
print "y_train", y_train.shape

X_test = test_array.reshape(len(test_files), -1)
y_test = test_targets
print "X_test", X_test.shape
print "y_test", y_test.shape

################################################################################################3
# Decision tree
clf = DecisionTreeClassifier()  # random_state=13)

clf.fit(X_train, y_train)

pred = clf.predict(X_test)

title = "Using plain vanilla Decision Tree accuracy is (%.3f)" % accuracy_score(y_test, pred)
print title
### Output:
###Using plain vanilla Decision Tree accuracy is (0.278)

#####################################
# Grid search to find best params
params = {'max_features': [64, 128, 256, 512], \
          'max_depth': [20, 50, 100], \
          'criterion': ['gini', 'entropy'] \
          }
clf = GridSearchCV(DecisionTreeClassifier(random_state=13), params, cv=5)

clf.fit(X_train, y_train)

print "Using Grid search:\n"
print "Best Params:", clf.best_params_
print

for params, mean_score, scores in clf.grid_scores_:
    print "%0.3f (+/- %0.03f) for %r" % (mean_score, scores.std() * 2, params)

##Output:
# Using Grid search:
# Best Params: {'max_features': 128, 'criterion': 'gini', 'max_depth': 20}
##0.384 (+/- 0.028) for {'max_features': 128, 'criterion': 'gini', 'max_depth': 20}

### Randomized search
##params = {    'max_features': sp_randint(10,1000), \
##              'max_depth': sp_randint(10,500), \
##              'criterion': ['gini','entropy'] \
##         }
##clf = RandomizedSearchCV(DecisionTreeClassifier(random_state=13), params, cv=5, n_iter=10)
##
##print "Best Params:", clf.best_params_
##print
##
##for params, mean_score, scores in clf.grid_scores_:
##    print "%0.3f (+/- %0.03f) for %r" % (mean_score, scores.std() * 2, params)

# Random forest
for n in range(5, 70, 10):
    clf = RandomForestClassifier(random_state=7, n_estimators=n)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    title = "Random Forest (%d): %.3f" % (n, accuracy_score(y_test, pred))
    print title

    # output
    ##Random Forest (5): 0.151
    ##Random Forest (15): 0.064
    ##Random Forest (25): 0.049
    ##Random Forest (35): 0.044
    ##Random Forest (45): 0.038
    ##Random Forest (55): 0.033
    ##Random Forest (65): 0.029