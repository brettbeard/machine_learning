import numpy as np
from sklearn import datasets
from sklearn.neural_network import MLPClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

# get the Iris dataset
iris = datasets.load_iris()

iris_X = iris.data
iris_y = iris.target

# Split iris data in train and test data
iris_X_train, iris_X_test, iris_y_train, iris_y_test = train_test_split(iris_X, iris_y, random_state=0)

# We set hidden_layer_size to (10) which means we add one hidden layer with 10 neurons.
# Set solver as 'sgd' because we will use Stochastic Gradient Descent as optimizer.
# Set learning_rate_init to 0.01, this is a learning rate value (be careful, don't
# confuse with alpha parameter in MLPClassifier).
# Then the last, we set 500 as the maximum number of training iteration.
mlp = MLPClassifier(hidden_layer_sizes=(10), solver='sgd', learning_rate_init=0.01, max_iter=500)
mlp.fit(iris_X_train, iris_y_train)

print("Training set score: %f" % mlp.score(iris_X_train, iris_y_train))
print("Test set score: %f" % mlp.score(iris_X_test, iris_y_test))

y_pred = mlp.predict(iris_X_test)

print confusion_matrix(iris_y_test, y_pred)




