# Heart-Attack-Prediction
Dataset: https://www.kaggle.com/datasets/rashikrahmanpritom/heart-attack-analysis-prediction-dataset

The dataset includes patients' age, sex, excercise induced angina, number of major vessels, chest pain type, resting blood pressure, cholesteral fetched via BMI sensor, fasting blood sugar, resting electrocardiaphic results and maximum heart rate achieved to predict if it has more chance of heart attack.

## Prediction without Framework
Created a simple neural network to predict a heart attack. It was implemented from scratch by using only python, numpy and pandas.
The network has one hidden layer. The parameters are updated after a pass of forward propagation and back propagation.

The dataset was split 80% for training and 20% for testing.
After executing the program, the accurracy is printed in console for each 10 epochs during training and at the end it prints the accuracy achieved for the test dataset.


## Prediction using Tensorflow
By using Tensorflow, the same neural network implemented from scratch was replecated. The model is trained for the same amount of epochs and displays its loss and accuracy for each one of them.

The dataset was again split 80% for training and 20% for testing. 20% of the training set was used as a validatioon set to display loss and accuracy obtained on each epoch.

After the model is trained, it plots loss and accuracy for the training set and the validation set. After this plots, the model is evaluated using the test set and prints the accuracy obtained.

The same was performed for an improved version of the neural network which uses different activation functions, initializers, optimizer and regularization techiques.

Both models can be compared since both of them print and plot their loss and accuracy obtained during training and test accuracy obtained using the same dataset split for training, validation and testing.

A more in depth analysis of these models is included in a pdf file in this same repository.
