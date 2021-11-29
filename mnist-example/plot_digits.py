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
from sklearn import tree
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sn

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
  x_train, x_test, y_train, y_test = train_test_split(data, targets, test_size=0.20, random_state=123)
  
  x_train, x_val, y_train, y_val   = train_test_split(x_train, y_train, test_size=0.50, random_state=123)

  return x_train, y_train, x_test, y_test, x_val, y_val

def split_train_dataset(x_train, y_train, test_size):
  x_training, x_testing, y_training, y_testing = train_test_split(x_train, y_train, test_size = test_size, random_state=123)
  return x_training, y_training

x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(data, digits.target)

test_acc_arr  = []
valid_acc_arr = []
train_acc_arr = []
f1_score_arr  = []
train_sample  = []
predictions   = []

for i in range (10):
  test_size = 1 - ((i + 1) / 10)

  if i == 9:
    x_training = x_train
    y_training = y_train
  else:
    x_training, y_training = split_train_dataset(x_train, y_train, test_size = test_size)

  train_sample.append(((i + 1) / 10) * 100)

  clf = tree.DecisionTreeClassifier()
  clf.fit(x_training, y_training)

  predicted      = clf.predict(x_val)
  valid_accuracy = metrics.accuracy_score(y_pred = predicted, y_true = y_val)
  valid_acc_arr.append(valid_accuracy)

  train_predict  = clf.predict(x_training)
  train_accuracy = metrics.accuracy_score(y_pred = train_predict, y_true = y_training)
  train_acc_arr.append(train_accuracy)

  test_predict  = clf.predict(x_test)

  test_accuracy = metrics.accuracy_score(y_pred = test_predict, y_true = y_test)
  f1_score      = metrics.f1_score(y_test, test_predict, average = 'macro')

  test_acc_arr.append(test_accuracy)
  f1_score_arr.append(f1_score)
  predictions.append(test_predict)

# Print the plot between training samples and f1_Score

x = train_sample
y = f1_score_arr

plt.figure(figsize=(8, 6))
plt.title('Training samples v/s f1 Score')
plt.xlabel('X axis - Training samples')
plt.ylabel('Y axis - f1 Score')
plt.xlim(0, 100)
plt.plot(x, y, color = 'blue')
plt.show()

print('From the graph above, we can see that the f1 score increases as we increase the training samples.')
print('This shows that adequate training samples are needed for the model to be trained properly so as to get good results')
print('\n')

# Print the accuracies and f1 score

print ("{:<15} {:<17} {:<30} {:<35} {:30}".format('Run', 'Train', 'Validation', 'Test', 'f1 Score'))
print('=============================================================================================================================')

for i in range (10):
  print ((i + 1), '\t \t', train_acc_arr[i], '\t \t', valid_acc_arr[i], '\t \t', test_acc_arr[i], '\t \t \t', f1_score_arr[i])

print('\n')
print('From the table above, we can see that accuracy is not always the best indicator for telling us about model performance.')
print('We need more parameters to accurately measure the model performance. For that we can plot the confusion matrix to get a better idea')
print('\n')

print('Printing the confusion matrices for 1 case')
print ('\n')

print('For training samples percentage as 10%')
cf_matrix = metrics.confusion_matrix(y_test, predictions[0])
sn.heatmap(cf_matrix, annot=True)
print ('\n')