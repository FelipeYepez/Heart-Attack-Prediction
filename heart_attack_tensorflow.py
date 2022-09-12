import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

heart_df = pd.read_csv('heart.csv', header = 0)

X = heart_df.drop(["output"], axis = 1).to_numpy()
Y = heart_df["output"].to_numpy()

scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

x_train, x_test, y_train, y_test=train_test_split(X, Y, test_size=0.2, random_state=43)

y_train = tf.one_hot(y_train, tf.constant(2))
y_test = tf.one_hot(y_test, tf.constant(2))
input_dim = x_train.shape[1]

# First Model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(input_dim,)))
model.add(tf.keras.layers.Dense(units=8, activation="relu", kernel_initializer='he_normal'))
model.add(tf.keras.layers.Dense(units=2, activation="softmax", kernel_initializer='he_normal'))

model.compile(
    optimizer=tf.keras.optimizers.SGD(
        learning_rate=0.01,
        momentum=0.0,
        name='SGD'
    ),
    loss="categorical_crossentropy",
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

print("Fit past model made without framework")
history = model.fit(x_train, y_train, epochs=400)

plt.subplot(1, 2, 1)
plt.title("Train")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history.history["loss"])
plt.show(block = False)

plt.subplot(1, 2, 2)
plt.title("Train")
plt.xlabel("Epochs")
plt.ylabel("Categorical Accuracy")
plt.plot(history.history["categorical_accuracy"])
plt.show()

evaluate = model.evaluate(x_test, y_test)
print("First Model Categorical Accuracy:", evaluate[1])


# Second improved Model
model2 = tf.keras.Sequential()
model2.add(tf.keras.Input(shape=(input_dim,)))
model2.add(tf.keras.layers.Dense(units=8, activation="tanh", kernel_initializer=tf.keras.initializers.HeUniform()))
model2.add(tf.keras.layers.Dropout(0.1))
model2.add(tf.keras.layers.Dense(units=6, activation="tanh", kernel_initializer=tf.keras.initializers.HeUniform()))
model2.add(tf.keras.layers.Dropout(0.1))
model2.add(tf.keras.layers.Dense(units=2, activation="softmax", kernel_initializer=tf.keras.initializers.HeUniform()))

model2.compile(
    optimizer="RMSprop",
    loss="categorical_crossentropy",
    metrics=[tf.keras.metrics.CategoricalAccuracy()]
)

print("Fit improved model")
batch_size = 32
history2 = model2.fit(x_train, y_train, 
    epochs=150, 
    steps_per_epoch=len(x_train)/batch_size, 
    verbose=2, 
    validation_data=(x_test, y_test)
)

plt.subplot(1, 2, 1)
plt.title("Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(history2.history["loss"], label="Train")
plt.plot(history2.history["val_loss"], label="Validation")
plt.legend()
plt.show(block = False)

plt.subplot(1, 2, 2)
plt.title("Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Categorical Accuracy")
plt.plot(history2.history["categorical_accuracy"], label="Train")
plt.plot(history2.history["val_categorical_accuracy"], label="Validation")
plt.legend()
plt.show()

evaluate2 = model2.evaluate(x_test, y_test)
print("Improved Model Categorical Test Accuracy:", evaluate2[1])