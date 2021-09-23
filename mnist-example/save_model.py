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
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean

import numpy as np

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

_, axes = plt.subplots(nrows=1, ncols=4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Training: %i' % label)

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

rescale_fac = [0.25, 0.5, 1, 2, 3]

for test_size, valid_size in [(0.15, 0.15), (0.20, 0.10)]:

  for rescale_factor in rescale_fac:
    model_list = []

    for gamma in [10 ** exponent for exponent in range(-7, 0)]:
      resized_img = []

      for itr1 in digits.images:
        resized_img.append(rescale(itr1, rescale_factor, anti_aliasing=True))
      
      resized_img = np.array(resized_img)
      data        = resized_img.reshape(n_samples, -1)

      clf = svm.SVC(gamma = gamma)

      x_train, x_test_valid, y_train, y_test_valid = train_test_split(data, digits.target, test_size = test_size + valid_size, shuffle=False)

      x_test, x_valid, y_test, y_valid             = train_test_split(x_test_valid, y_test_valid, test_size = valid_size / (test_size + valid_size), shuffle=False)

      clf.fit(x_train, y_train)

      pred_valid = clf.predict(x_valid)
      acc_valid  = metrics.accuracy_score(y_pred = pred_valid, y_true = y_valid)
      valid_f1   = metrics.f1_score(y_pred = pred_valid, y_true = y_valid, average = 'macro')

      if acc_valid < 0.11:
        print('Skipping for {}'.format(gamma))
        continue
        
      test_model = {'model': clf, 'Validation accuracy': acc_valid, 'F1 score': valid_f1, 'Gamma': gamma,}

      model_list.append(test_model)

    best_model = max(model_list, key = lambda x: x['F1 score'])

    prediction = best_model['model'].predict(x_test)
    accuracy   = metrics.accuracy_score(y_pred = prediction, y_true = y_test)
    f1_score   = metrics.f1_score(y_pred = prediction, y_true = y_test, average = 'macro')

    print('{} * {} \t {} : {} \t {:.3f} \t {:.3f}'.format(resized_img[0].shape[0], resized_img[0].shape[1], best_model['Gamma'], (1 - test_size) * 100, test_size * 100,
                                                          accuracy, f1_score))