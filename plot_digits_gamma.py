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

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Split the data into test and validation data
x_test, x_valid, y_test, y_valid = train_test_split(X_test, y_test, test_size = 0.7, shuffle=True)

best_acc   = 0
best_gamma = 0

test_acc_arr  = []
valid_acc_arr = []
gamma = []

gamma_val = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10]

print ("{:<20} {:<20} {:<40}".format('Gamma Value', 'Validation Accuracy', 'Test Accuracy'))

for val in gamma_val:
  gamma.append(val)
  clf       = svm.SVC(gamma = val)
  clf.fit(X_train, y_train)
  predicted = clf.predict(x_valid)
  
  valid_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_valid)
  valid_acc_arr.append(valid_accuracy)
  
  valid_f1       = metrics.f1_score(y_pred = predicted, y_true = y_valid, average = 'macro')

  test_predict   = clf.predict(x_test)
  test_accuracy = metrics.accuracy_score(y_pred = test_predict, y_true = y_test)

  print(val, '\t \t', valid_accuracy, '\t \t \t', test_accuracy)

best_acc = max(valid_acc_arr)

itr1 = 0

for itr1 in range(0, len(valid_acc_arr)):
  if best_acc == valid_acc_arr[itr1]:
    best_gamma = gamma[itr1]

print('The best gamma value is:', best_gamma)
print('The best accuracy score is:', best_acc)