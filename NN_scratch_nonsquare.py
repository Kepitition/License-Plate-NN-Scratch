import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle

# Load and preprocess the dataset
# data = pd.read_csv('license_plate_28_56_data.csv')
# data = np.array(data)
# m, n = data.shape
# np.random.shuffle(data)
#
# # Split into training and development sets
# data_dev = data[0:1000].T
# Y_dev = data_dev[0]
# X_dev = data_dev[1:n] / 255.
#
# data_train = data[1000:m].T
# Y_train = data_train[0]
# X_train = data_train[1:n] / 255.
# _, m_train = X_train.shapee


def init_params():
    W1 = np.random.normal(size=(128, 1568)) * np.sqrt(1. / 1568)  # Adjusted for 28x56 input size
    b1 = np.zeros((128, 1))
    W2 = np.random.normal(size=(64, 128)) * np.sqrt(1. / 128)
    b2 = np.zeros((64, 1))
    W3 = np.random.normal(size=(35, 64)) * np.sqrt(1. / 64)  # Number of classes
    b3 = np.zeros((35, 1))  # Number of classes
    return W1, b1, W2, b2, W3, b3


def ReLU(Z):
    return np.maximum(Z, 0)


def ReLU_deriv(Z):
    return Z > 0


def softmax(Z):
    Z -= np.max(Z, axis=0)
    A = np.exp(Z) / np.sum(np.exp(Z), axis=0)
    return A


def forward_prop(W1, b1, W2, b2, W3, b3, X):
    Z1 = W1.dot(X) + b1
    A1 = ReLU(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = ReLU(Z2)
    Z3 = W3.dot(A2) + b3
    A3 = softmax(Z3)
    return Z1, A1, Z2, A2, Z3, A3


def one_hot(Y):
    NUM_CLASSES = 35
    one_hot_Y = np.zeros((Y.size, NUM_CLASSES))
    one_hot_Y[np.arange(Y.size), Y] = 1
    return one_hot_Y.T


def backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y):
    one_hot_Y = one_hot(Y)
    dZ3 = A3 - one_hot_Y
    dW3 = 1 / m_train * dZ3.dot(A2.T)
    db3 = 1 / m_train * np.sum(dZ3, axis=1, keepdims=True)

    dZ2 = W3.T.dot(dZ3) * ReLU_deriv(Z2)
    dW2 = 1 / m_train * dZ2.dot(A1.T)
    db2 = 1 / m_train * np.sum(dZ2, axis=1, keepdims=True)

    dZ1 = W2.T.dot(dZ2) * ReLU_deriv(Z1)
    dW1 = 1 / m_train * dZ1.dot(X.T)
    db1 = 1 / m_train * np.sum(dZ1, axis=1, keepdims=True)

    return dW1, db1, dW2, db2, dW3, db3


def update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha):
    W1 -= alpha * dW1
    b1 -= alpha * db1
    W2 -= alpha * dW2
    b2 -= alpha * db2
    W3 -= alpha * dW3
    b3 -= alpha * db3
    return W1, b1, W2, b2, W3, b3


def get_predictions(A):
    return np.argmax(A, axis=0)


def get_accuracy(predictions, Y):
    return np.sum(predictions == Y) / Y.size


def gradient_descent(X, Y, alpha, iterations):
    W1, b1, W2, b2, W3, b3 = init_params()
    for i in range(iterations):
        Z1, A1, Z2, A2, Z3, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
        dW1, db1, dW2, db2, dW3, db3 = backward_prop(Z1, A1, Z2, A2, Z3, A3, W1, W2, W3, X, Y)
        W1, b1, W2, b2, W3, b3 = update_params(W1, b1, W2, b2, W3, b3, dW1, db1, dW2, db2, dW3, db3, alpha)
        if i % 10 == 0:
            predictions = get_predictions(A3)
            print(f"Iteration {i}: Accuracy = {get_accuracy(predictions, Y)}")
    return W1, b1, W2, b2, W3, b3


def make_predictions(X, W1, b1, W2, b2, W3, b3):
    _, _, _, _, _, A3 = forward_prop(W1, b1, W2, b2, W3, b3, X)
    return get_predictions(A3)


def test_prediction(index, W1, b1, W2, b2, W3, b3):
    current_image = X_train[:, index, None]
    prediction = make_predictions(current_image, W1, b1, W2, b2, W3, b3)
    label = Y_train[index]
    print(f"Prediction: {prediction[0]}, Label: {label}")

    current_image = current_image.reshape((56, 28)) * 255  # Adjusted for 28x56
    plt.gray()
    plt.imshow(current_image, interpolation='nearest')
    plt.show()

def save_model_parameters(filename, W1, b1, W2, b2, W3, b3):
    parameters = {"W1": W1, "b1": b1, "W2": W2, "b2": b2, "W3": W3, "b3": b3}
    with open(filename, 'wb') as f:
        pickle.dump(parameters, f)
    print(f"Model parameters saved to {filename}")


def load_model_parameters(filename):
    with open(filename, 'rb') as f:
        parameters = pickle.load(f)
    print(f"Model parameters loaded from {filename}")
    return parameters["W1"], parameters["b1"], parameters["W2"], parameters["b2"], parameters["W3"], parameters["b3"]