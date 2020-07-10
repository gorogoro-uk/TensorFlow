# MNIST
# convolution & max pooling

import tensorflow as tf



# data
mnist_data = tf.keras.datasets.mnist
(train_images, train_labels),(test_images, test_labels) = mnist_data.load_data()
train_images = train_images.reshape(60000, 28, 28, 1)
test_images = test_images.reshape(10000, 28, 28, 1)
train_images= train_images/255.0
test_images = test_images/255.0

# accuracy callback
ACCURACY_LIMIT = 0.995
class accCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > ACCURACY_LIMIT):
            print("\nTraining terminated: reached desired accuracy {ACCURACY_LIMIT*100}%\n")
            self.model.stop_training = True

mnist_acc_callback = accCallBack()

# model definition
mnist_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), input_shape=(28, 28, 1), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(500, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax'),
])
mnist_model.summary()

# compile model
mnist_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# fit model
mnist_model.fit(train_images, train_labels, epochs=10, callbacks=[mnist_acc_callback])