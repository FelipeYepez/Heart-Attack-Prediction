import numpy as np
import pandas as pd

heart_df = pd.read_csv('heart.csv', header = 0)
# Shuffle dataframe
heart_df = heart_df.sample(frac=1)

n_train = heart_df.shape[0] * 0.8

Y = heart_df["output"].to_numpy()

def normalize(series):
    return series / series.abs().max()
    
for col in heart_df.columns:
    heart_df[col] = normalize(heart_df[col])

X = heart_df.drop(["output"], axis = 1).to_numpy()

train_X = X[0: int(n_train), :].T
test_X = X[int(n_train):, :].T

train_Y = Y[0: int(n_train), ].T
test_Y =  Y[int(n_train):, ].T



def init_params_he(layer1, layer2):
    W1 = np.random.randn(layer1, train_X.shape[0]) * np.sqrt(2 / train_X.shape[0], dtype='float64')
    b1 = np.zeros((layer1, 1), dtype='float64')
    W2 = np.random.randn(layer2, layer1) * np.sqrt(2 / layer1, dtype='float64')
    b2 = np.zeros((layer2, 1), dtype='float64')
    return W1, b1, W2, b2

def ReLU(Z):
    return np.maximum(Z, 0)

def softmax(Z):
    A = np.exp(Z) / sum(np.exp(Z + 1e-18))
    return A
    
def forward_prop(W1, b1, W2, b2, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def ReLU_deriv(Z):
    return Z > 0

def one_hot(Y):
    one_hot_Y = np.zeros((Y.size, int(Y.max()) + 1))
    one_hot_Y[np.arange(Y.size), Y] = 1
    one_hot_Y = one_hot_Y.T
    return one_hot_Y

def backward_prop(Z1, A1, Z2, A2, W1, W2, X, one_hot_Y):
    m = X.shape[1]
    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)
    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate):
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1    
    W2 = W2 - learning_rate * dW2  
    b2 = b2 - learning_rate * db2    
    return W1, b1, W2, b2


def get_predictions(A2):
    return np.rint(A2)

def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, learning_rate, epochs, hidden_neurons):
    classifications = 2
    W1, b1, W2, b2 = init_params_he(hidden_neurons, classifications)
    for i in range(epochs):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward_prop(Z1, A1, Z2, A2, W1, W2, X, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, learning_rate)
        if i % 10 == 0:
            print("Iteration: ", i)
            predictions = get_predictions(A2)
            print("Train accuracy: ", get_accuracy(predictions, Y))
    return W1, b1, W2, b2

W1, b1, W2, b2 = gradient_descent(train_X, one_hot(train_Y), learning_rate = 0.10, epochs = 400, hidden_neurons = 8)


def test(W1, b1, W2, b2, X, Y):
    Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, X)
    predictions = get_predictions(A2)
    print("Test accuracy: ", get_accuracy(predictions, Y))

test(W1, b1, W2, b2, test_X, one_hot(test_Y))