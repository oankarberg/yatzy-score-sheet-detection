import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt


mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()


y_train = y_train.astype('float32')
y_test = y_test.astype('float32')


# Create binary image
mask = x_train > 127.5
maskb = x_train <= 127.5
x_train[mask] = 255
x_train[maskb] = 0

# Create binary image
mask_test = x_test > 127.5
mask_test_b = x_test <= 127.5
x_test[mask_test] = 255
x_test[mask_test_b] = 0

# Shift to -1 to 1
x_train, x_test = (x_train - 127.5) / 127.5, (x_test - 127.5) / 127.5


# Reserve 10,000 samples for validation
x_val = x_train[-10000:]
y_val = y_train[-10000:]
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# Simple CNN architecture with 2 conv layers
# INPUT -> CONV3-32 -> RELU -> CONV3-32 -> RELU  -> CONV3-32 -> RELU -> POOL -> DROPOUT -> FC -> RELU -> DROPOUT -> SOFTMAX
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                           activation='relu',
                           input_shape=(28, 28, 1)),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                           activation='relu'),
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3),
                           activation='relu'),
    # regularization
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),
    # regularization
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    # regularization
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], x_test.shape[2], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))


model_data = model.fit(x_train, y_train, epochs=10, batch_size=128, validation_data=(x_val, y_val))
model.evaluate(x_test, y_test, verbose=2)

# plot
plt.subplot(2, 1, 1)
plt.plot(model_data.history['accuracy'])
plt.plot(model_data.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train acc', 'val acc'], loc='lower right')
plt.subplot(2, 1, 2)
plt.plot(model_data.history['loss'])
plt.plot(model_data.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train loss', 'val loss'], loc='upper right')
plt.tight_layout()
plt.show()


model.save('./models/model_tensorflow')
