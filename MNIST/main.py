import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping

# Create a simple Keras model with a convolutional layer.
model = tf.keras.Sequential([
    # Convolutional layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    
    # Dense layers
    layers.Dense(64, activation='relu'),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # One neuron for each class 0,1,2,3,...,9
])

# Compile the model.
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',  # Use sparse_categorical_crossentropy
              metrics=['accuracy'])

# Load the MNIST dataset, a well-known dataset of handwritten digits.
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Reshape the images for the convolutional layer
x_train = x_train.reshape(-1, 28, 28, 1)  # Reshape to (60000, 28, 28, 1)
x_test = x_test.reshape(-1, 28, 28, 1)    # Reshape to (10000, 28, 28, 1)

# Normalize the images
x_train, x_test = x_train / 255.0, x_test / 255.0  # Normalize the data

# Define the early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
model.fit(x_train, y_train, epochs=50, validation_split=0.1, callbacks=[early_stopping])

# Evaluate the model
model.evaluate(x_test, y_test)

# Save the model
model.save('MNIST/mnist_model.h5')