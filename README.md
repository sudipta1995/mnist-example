This repo has been created using the code from https://scikit-learn.org/stable/auto_examples/classification/plot_digits_classification.html#sphx-glr-auto-examples-classification-plot-digits-classification-py

The main branch has the Readme and requirements file.

The branch - feature/plots has the actual base code that recognizes the handwritten digits.

Another branch - plot/gamma deals with the hyperparameter tuning of the code.
It consists of 3 files - plot_graph_gamma1.py, plot_graph_gamma2.py, plot_graph_gamma3.py. Each file has a different value of gamma (done as part of hyperparameter tuning) and the results obtained from all 3 variations can be seen below.

![Gamma_value1_results](https://user-images.githubusercontent.com/76610555/132897943-e90fa201-1f28-48c7-b306-2b00cf9cd2a4.PNG)
![Gamma_value2_results](https://user-images.githubusercontent.com/76610555/132897952-e2edfb8a-ded3-4d47-a22b-6bdb01cfeda2.PNG)
![Gamma_value3_results](https://user-images.githubusercontent.com/76610555/132897953-86ac7354-5a4f-41f5-bbba-309a3809d8fd.PNG)
![Hyperparameter_tuning_results](https://user-images.githubusercontent.com/76610555/132897955-38d09b33-1423-4624-bd37-416f2f69db9e.PNG)

From the results obtained, its pretty evident that changing the gamma value directly impacts the results (accuracy, f1 score, precision). As the value of gamma increases, the performance decreases. From 97% accuracy for gamma = 0.001, the accuracy drops to ~10% for gamma=0.05.

Hence, the value of gamma is inversely proportional to the accuracy of the digit prediction.
