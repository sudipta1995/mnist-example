"""
================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.
"""

print(__doc__)

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# License: BSD 3 clause

# Standard scientific Python imports
import matplotlib.pyplot as plt

# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

###############################################################################
# Digits dataset
# --------------
#
# The digits dataset consists of 8x8
# pixel images of digits. The ``images`` attribute of the dataset stores
# 8x8 arrays of grayscale values for each image. We will use these arrays to
# visualize the first 4 images. The ``target`` attribute of the dataset stores
# the digit each image represents and this is included in the title of the 4
# plots below.
#
# Note: if we were working from image files (e.g., 'png' files), we would load
# them using :func:`matplotlib.pyplot.imread`.

digits = datasets.load_digits()

###############################################################################
# Classification
# --------------
#
# To apply a classifier on this data, we need to flatten the images, turning
# each 2-D array of grayscale values from shape ``(8, 8)`` into shape
# ``(64,)``. Subsequently, the entire dataset will be of shape
# ``(n_samples, n_features)``, where ``n_samples`` is the number of images and
# ``n_features`` is the total number of pixels in each image.
#
# We can then split the data into train and test subsets and fit a support
# vector classifier on the train samples. The fitted classifier can
# subsequently be used to predict the value of the digit for the samples
# in the test subset.

# flatten the images
n_samples = len(digits.images)
data      = digits.images.reshape((n_samples, -1))


def split_dataset(data, targets):
  x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.15, random_state=123)
  
  x_train, x_val, y_train, y_val   = train_test_split(x_train, y_train, test_size=0.15, random_state=123)

  return x_train, y_train, x_test, y_test, x_val, y_val

best_acc   = 0
best_gamma = 0

test_acc_arr  = []
valid_acc_arr = []
train_acc_arr = []
mean_arr      = []

x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(data, digits.target)

clf       = MLPClassifier(alpha=1, max_iter=2)

clf.fit(x_train, y_train)

predicted = clf.predict(x_val)
valid_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_val)
valid_acc_arr.append(valid_accuracy)

train_predict = clf.predict(x_train)
train_accuracy = metrics.accuracy_score(y_pred = train_predict, y_true = y_train)
train_acc_arr.append(train_accuracy)

test_predict   = clf.predict(x_test)
test_accuracy = metrics.accuracy_score(y_pred = test_predict, y_true = y_test)
test_acc_arr.append(test_accuracy)

mean = ((valid_accuracy + train_accuracy + test_accuracy) / 3)
mean_arr.append(mean)

print ("{:<20} {:<20} {:<30}".format('Train', 'Dev', 'Test'))
print('-----------------------------------------------')

print (train_acc_arr[0], '\t', valid_acc_arr[0], '\t', test_acc_arr[0])