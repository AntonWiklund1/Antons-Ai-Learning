import tensorflow as tf
from tensorflow.keras import layers


# Create a simple Keras model.
model = tf.keras.Sequential([
    # Add a simple dense layer.
    layers.Dense(64, activation='relu', input_shape=(784,)),  # Correct input shape
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax') # One neuron for each class 0,1,2,3,...,9
])


# Compile the model. 
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy
              metrics=['accuracy'])

# Load the MNIST dataset, a well-known dataset of handwritten digits.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the images to a flat vector
x_train = x_train.reshape(-1, 784)  # Reshape from (60000, 28, 28) to (60000, 784)
x_test = x_test.reshape(-1, 784)  # Reshape from (10000, 28, 28) to (10000, 784)

# Normalize the images
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data

# Train the model
model.fit(x_train, y_train, epochs=10)

# Evaluate the model
model.evaluate(x_test, y_test)