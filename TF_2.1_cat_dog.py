# Cats - Dogs image classification

# import code
import tensorflow as tf
import os
import urllib.request
import zipfile
import random
from shutil import copyfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# prepare data directories
DATA_DEST = '/catdog'                       # top directory
ZIPFILE_DEST = '/catdog/cats-and-dogs.zip'  # zip file download file
CAT_SOURCE = '/catdog/PetImages/Cat/'       # unzipped cat images
DOG_SOURCE = '/catdog/PetImages/Dog/'       # unzipped dog images
IMAGE_DEST = '/catdog/image'                 # image directory
TRAIN_DEST = '/catdog/image/train'           # training images
TEST_DEST = '/catdog/image/test'             # testing images
TRAIN_CAT = '/catdog/image/train/cat/'       # cat training images
TEST_CAT = '/catdog/image/test/cat/'         # cat testing images
TRAIN_DOG = '/catdog/image/train/dog/'       # dog training images
TEST_DOG = '/catdog/image/test/dog/'         # dog testing images
"""
# create directories
try:
    os.mkdir(DATA_DEST)
    os.mkdir(TRAIN_DEST)
    os.mkdir(TEST_DEST)
    os.mkdir(TRAIN_CAT)
    os.mkdir(TRAIN_DOG)
    os.mkdir(TEST_CAT)
    os.mkdir(TEST_DOG)
except OSError:
    pass
"""
# download data
URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
urllib.request.urlretrieve(URL, ZIPFILE_DEST)

# unzip data
zip_ref = zipfile.ZipFile(ZIPFILE_DEST, 'r')
zip_ref.extractall(DATA_DEST)
zip_ref.close()

# confirm download
print(len(os.listdir(CAT_SOURCE)))
print(len(os.listdir(DOG_SOURCE)))
# Expected Output:
# 12501
# 12501


# split data into train & test
TRAIN_SIZE = 0.90


def split_data(source, train, test, split):
    files = []
    for filename in os.listdir(source):
        file = source + filename
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    train_length = int(len(files) * split)
    test_length = int(len(files) - train_length)
    shuffled_set = random.sample(files, len(files))
    train_set = shuffled_set[0:train_length]
    test_set = shuffled_set[-test_length:]

    for filename in train_set:
        this_file = source + filename
        destination = train + filename
        copyfile(this_file, destination)

    for filename in test_set:
        this_file = source + filename
        destination = test + filename
        copyfile(this_file, destination)


# split cat & dog data
split_data(CAT_SOURCE, TRAIN_CAT, TEST_CAT, TRAIN_SIZE)
split_data(DOG_SOURCE, TRAIN_DOG, TEST_DOG, TRAIN_SIZE)

# confirm train/test split
print(len(os.listdir(TRAIN_CAT)))
print(len(os.listdir(TRAIN_DOG)))
print(len(os.listdir(TEST_CAT)))
print(len(os.listdir(TEST_DOG)))
# Expected output:
# 11250
# 11250
# 1250
# 1250


# data generator
train_image_datagen = ImageDataGenerator(rescale=1.0/255.)
train_datagen = train_image_datagen.flow_from_directory(TRAIN_DEST,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


test_image_datagen = ImageDataGenerator(rescale=1.0/255.)
test_datagen = test_image_datagen.flow_from_directory(TEST_DEST,
                                                    batch_size=100,
                                                    class_mode='binary',
                                                    target_size=(150, 150))


# accuracy callback
ACCURACY_LIMIT = 0.99
class accCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > ACCURACY_LIMIT):
            print(f"Training terminated: reached {ACCURACY_LIMIT*100} accuracy level.")
            self.model.stop_training=True

acc_limit_callback = accCallback()

# model definition
catdog_model = tf.keras.models.Sequential(
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_size=(150,150,3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
)
catdog_model.summary()

# compile model
catdog_model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)

# fit model
catdog_history = catdog_model.fit(
    train_datagen,
    epochs=10,
    callbacks=[acc_limit_callback],
    verbose=1,
    validation_data=test_datagen
)


# plot results
import matplotlib.pyplot as plt

train_acc = catdog_history.history['accuracy']
test_acc = catdog_model.history['val_accuracy']
train_loss = catdog_history.history['loss']
test_loss = catdog_history.history['val_loss']
epochs = range(len(train_acc))

# plot accuracy per epoch
plt.plot(epochs, train_acc, 'r', 'Training Accuracy')
plt.plot(epochs, test_acc, 'b', 'Testing Accuracy')
plt.title('Accuracy')
plt.figure()

# plot loss per epoch
plt.plot(epochs, train_acc, 'r', 'Training Loss')
plt.plot(epochs, test_acc, 'b', 'Testing Loss')
plt.title('Loss')
plt.figure()
