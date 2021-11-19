import os
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from skimage import data, color
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split
import pickle
from flask import Flask, request
import numpy as np
import sys

app = Flask(__name__)
sys.path.insert(1, '/home/sudipta/mlops/mnist-example/mnist-example/tests/)
import utils

from utils import preprocess, create_splits, test

clf = utils.load('/home/sudipta/mlops/mnist-example/mnist-example/tests/SVM_model.pkl')

clf1 = utils.load('/home/sudipta/mlops/mnist-example/mnist-example/tests/DT_model.pkl')

def model_load_svm(clf):
    digits         = datasets.load_digits()
    n_samples      = len(digits.images)
    data           = digits.images.reshape((n_samples, -1))
    rescale_factor = 1

    x_train, x_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.3, shuffle=False)

    clf.fit(x_train, y_train)

    predicted      = clf.predict(x_test)

    print(f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_test, predicted)}\n")

	disp = metrics.plot_confusion_matrix(clf, x_test, y_test)
	disp.figure_.suptitle("Confusion Matrix")
	print(f"Confusion matrix:\n{disp.confusion_matrix}")

	acc = metrics.accuracy_score(y_test, predicted)
	f1  = metrics.f1_score(y_test, predicted, average = 'macro')
	
	plt.show()

	return acc, f1

def model_load_dt(clf1):
    digits1         = datasets.load_digits()
    n_samples1      = len(digits1.images)
    data1           = digits1.images.reshape((n_samples1, -1))
    rescale_factor1 = 1

    x_train1, x_test1, y_train1, y_test1 = train_test_split(data, digits1.target, test_size=0.3, shuffle=False)

    clf1.fit(x_train, y_train)

    predicted1      = clf.predict(x_test1)

    print(f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_test1, predicted1)}\n")

	disp1 = metrics.plot_confusion_matrix(clf1, x_test1, y_test1)
	disp1.figure_.suptitle("Confusion Matrix")
	print(f"Confusion matrix:\n{disp1.confusion_matrix}")

	acc1  = metrics.accuracy_score(y_test1, predicted1)
	f1_1  = metrics.f1_score(y_test1, predicted1, average = 'macro')
	
	plt.show()

	return acc1, f1_1

# Test Case 1 - Test SVM model

def test_svm_model():
    acc, f1 = model_load_svm(clf)
    assert acc  > 0.80
	assert f1   > 0.8

# Test Case 2 - Test DT model

def test_dt_model():
    acc1, f1_1  = model_load_dt(clf1)
    assert acc  > 0.80
	assert f1   > 0.8