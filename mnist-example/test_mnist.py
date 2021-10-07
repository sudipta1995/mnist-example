import os
import matplotlib.pyplot as plt
from sklearn import datasets, svm, metrics
from skimage import data, color
import numpy as np
from joblib import dump, load
from sklearn.model_selection import train_test_split

#from utils import preprocess, create_splits, test

digits = datasets.load_digits()

# flatten the images
n_samples      = len(digits.images)
data           = digits.images.reshape((n_samples, -1))
rescale_factor = 1

output_folder  = ''

# Split data into 50% train and 50% test subsets
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

def run_classification_experiment(X_train, X_test, y_train, y_test, output_folder):

	clf = svm.SVC(gamma=0.001)

	# Learn the digits on the train subset
	clf.fit(X_train, y_train)

	# Predict the value of the digit on the test subset
	predicted = clf.predict(X_test)

	test_size_disp  = 0.5
	train_size_disp = 0.5
	gamma_disp      = 0.001

	output_folder = "/home/sudipta/mlops/model_save_{}_val_{}_rescale_{}_gamma_{}".format(test_size_disp, train_size_disp, rescale_factor, gamma_disp)
	os.mkdir(output_folder)
	dump(clf, os.path.join(output_folder, "model.joblib"))

	return output_folder

def check_metrics(X_train, X_test, y_train, y_test):

	clf = svm.SVC(gamma=0.001)

	# Learn the digits on the train subset
	clf.fit(X_train, y_train)

	# Predict the value of the digit on the test subset
	predicted = clf.predict(X_test)

	print(f"Classification report for classifier {clf}:\n" f"{metrics.classification_report(y_test, predicted)}\n")

	disp = metrics.plot_confusion_matrix(clf, X_test, y_test)
	disp.figure_.suptitle("Confusion Matrix")
	print(f"Confusion matrix:\n{disp.confusion_matrix}")

	acc = metrics.accuracy_score(y_test, predicted)
	f1  = metrics.f1_score(y_test, predicted, average = 'macro')
	
	plt.show()

	return acc, f1

# Test Case 1 - Model creation

def test_model_writing():
	run_classification_experiment(X_train, X_test, y_train, y_test, output_fold##er)
	assert os.path.isfile(output_folder)

# Test Case 2 - Metrics testing

data_train, data_test, labels_train, labels_test = train_test_split(X_train, y_train, test_size=0.5, shuffle=False)

def test_small_data_overfit_checking():
	acc, f1 = check_metrics(data_train, data_test, labels_train, labels_test)
	assert acc  > 0.80
	assert f1   > 0.80