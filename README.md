This repo has been created using the code from https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

The main branch has the Readme and requirements file.

The branch - feature/plot has the base code that recognizes the handwritten digits.

Another branch - plots/gamma_comp deals with the hyperparameter tuning of the code. The SVM calssifier has been tested on various values of gamma and the value for which the model gives best accuracy has been computed.
It consists of 3 files - plot_graph_gamma1.py, plot_graph_gamma2.py, plot_graph_gamma3.py. Each file has a different value of gamma (done as part of hyperparameter tuning) and the results obtained from can be seen below.

![Result](https://user-images.githubusercontent.com/76610555/133898016-3f7e04e5-810e-4cff-9ed5-5a9bb13add23.PNG)

From the results obtained, its pretty evident that changing the gamma value directly impacts the results (accuracy, f1 score, precision). As the value of gamma increases, the performance decreases. From 97% accuracy for gamma = 0.001, the accuracy drops to ~10% for gamma=0.05.

Hence, the value of gamma is inversely proportional to the accuracy of the digit prediction.
