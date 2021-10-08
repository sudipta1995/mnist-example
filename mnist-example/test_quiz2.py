import os
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from skimage import data, color
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split

from utils import preprocess, create_splits

digits = datasets.load_digits()

# flatten the images
n_samples      = len(digits.images)
data           = digits.images.reshape((n_samples, -1))
rescale_factor = 1

data1  = data
target = digits.target

def test_create_splits():
  X_train, X_test, X_valid, y_train, y_test, y_valid = create_splits(data1, target, train_size=0.70, test_size=0.20, valid_size=0.10)

  assert (len(X_train) + len(X_test) + len(X_valid) + len(y_train) + len(y_test) + len(y_valid) == len(data1) + len(target))
  assert (len(X_train) + len(y_train)) == 0.70 * (len(data1) + len(target))
  assert (len(X_test) + len(y_test)) == 0.20 * (len(data1) + len(target))
  assert (len(X_valid) + len(y_valid)) == 0.10 * (len(data1) + len(target))