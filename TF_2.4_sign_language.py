# sign language classification
# multi-class

import tensorflow as tf
import os
from pathlib import Path
import csv
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# get data
# read in csv files (train & test), convert to numpy
# first column is label, then 784 columns of pixel values (28x28)
#os.chdir('/Users/seanmulligan/PycharmProjects/TF_pratice/')
CWD = Path(os.getcwd())
TRAIN_FILE = CWD / 'signlanguage/sign_mnist_train.csv'
TEST_FILE = CWD / 'signlanguage/sign_mnist_test.csv'
print("CWD:        ",CWD)
print("TRAIN FILE: ",TRAIN_FILE)
print("TEST FILE:  ",TEST_FILE)

def prepare_data(file_name):
    with open(file_name) as data_file:
        csv_reader = csv.reader(data_file, delimiter=',')
        first_line = True
        label_temp = []
        image_temp = []
        for row in csv_reader:
            if first_line:
                first_line = False
            else:
                label_temp.append(row[0])
                image_row = row[1:785]
                image_array = np.reshape(image_row, (28,28))
                image_temp.append(image_array)
    labels = np.array(label_temp).astype('float')
    images = np.array(image_temp).astype('float')
    return images, labels

x_train, y_train = prepare_data(TRAIN_FILE)
x_test, y_test = prepare_data(TEST_FILE)
print("train size: ",x_train.shape, y_train.shape)
print("test size:  ",x_test.shape, y_test.shape)

x_train = np.expand_dims(x_train, axis=3)
x_test = np.expand_dims(x_test, axis=3)

print("train size: ",x_train.shape, y_train.shape)
print("test size:  ",x_test.shape, y_test.shape)
print("\ndata prep ok\n")

# data generators - augmentation, normalisation
train_image_datagen = ImageDataGenerator(rescale=1/255.,
                                    rotation_range=40,
                                    width_shift_range=0.2,
                                    height_shift_range=0.2,
                                    shear_range=0.2,
                                    zoom_range=0.2,
                                    horizontal_flip=True,
                                    fill_mode='nearest'
)
train_datagen = train_image_datagen.flow(x_train, y_train, batch_size=32)

test_image_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = test_image_datagen.flow(x_test, y_test, batch_size=32)
print("\ndata generators ok\n")

# define model
sign_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3,3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(26, activation='softmax')]
)
sign_model.summary()
sign_model.compile(optimizer='Adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print("\nmodel ok\n")

# fit model
sign_model_history = sign_model.fit(train_datagen, epochs=10, verbose=1, validation_data=test_datagen)

print("\nfit model ok\n")

# learning
train_acc = sign_model_history.history['accuracy']
test_acc = sign_model_history.history['val_accuracy']
train_loss = sign_model_history.history['loss']
test_loss = sign_model_history.history['val_loss']
num_epochs = range(len(train_acc))

plt.plot(num_epochs, train_acc, 'r', label='train accuracy')
plt.plot(num_epochs, test_acc, 'b', label='test accuracy')
plt.title("Accuracy")
plt.legend()
plt.figure()

plt.plot(num_epochs, train_loss, 'r', label='train loss')
plt.plot(num_epochs, test_loss, 'b', label='test loss')
plt.title("Loss")
plt.legend()
plt.figure()
print("\ncharts ok\n")
plt.show()

