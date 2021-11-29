================================
Recognizing hand-written digits
================================
This example shows how scikit-learn can be used to recognize images of
hand-written digits, from 0-9.

![f1 score vs training sample graph](https://user-images.githubusercontent.com/76610555/143929283-ba427755-1956-4f97-9291-d59eb3133d69.PNG)

From the graph above, we can see that the f1 score increases as we increase the training samples.
This shows that adequate training samples are needed for the model to be trained properly so as to get good results

![Performance metrics](https://user-images.githubusercontent.com/76610555/143929312-d31de9e5-e3be-4ffd-b2ba-8bff141b1111.PNG)

From the table above, we can see that accuracy is not always the best indicator for telling us about model performance.
We need more parameters to accurately measure the model performance. For that we can plot the confusion matrix to get a better idea

Printing the confusion matrix for 1 case for training samples percentage as 10%

![Confusion matrix for 10%](https://user-images.githubusercontent.com/76610555/143929355-d5e9c9d5-55bf-4261-987a-688dccbc3cff.PNG)
