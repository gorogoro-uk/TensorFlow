# Cats - Dogs image classification

# import code
import tensorflow as tf
from pathlib import Path
import os
import urllib.request
import zipfile
import random
from shutil import copyfile
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# prepare data directories
BASE = Path(os.getcwd()) / 'catdog'   # base directory
ZIP_DEST = BASE / 'cats_dogs.zip'     # zip file destination
CAT_SOURCE = BASE / 'PetImages/Cat'
DOG_SOURCE = BASE / 'PetImages/Dog'
TRAIN_DEST = BASE / 'train'           # training images
TEST_DEST = BASE / 'test'             # testing images
TRAIN_CAT = BASE / 'train/cat'       # cat training images
TEST_CAT = BASE / 'test/cat'         # cat testing images
TRAIN_DOG = BASE / 'train/dog'       # dog training images
TEST_DOG = BASE / 'test/dog'         # dog testing images

print(BASE)
print(ZIP_DEST)
print(CAT_SOURCE)
print(DOG_SOURCE)
print(TRAIN_DEST)
print(TEST_DEST)
print(TRAIN_CAT)
print(TEST_CAT)
print(TRAIN_DOG)
print(TEST_DOG)

# create directories
if not os.path.exists(BASE):
    os.mkdir(BASE)
if not os.path.exists(TRAIN_DEST):
    os.mkdir(TRAIN_DEST)
if not os.path.exists(TEST_DEST):
    os.mkdir(TEST_DEST)
if not os.path.exists(TRAIN_CAT):
    os.mkdir(TRAIN_CAT)
if not os.path.exists(TRAIN_DOG):
    os.mkdir(TRAIN_DOG)
if not os.path.exists(TEST_CAT):
    os.mkdir(TEST_CAT)
if not os.path.exists(TEST_DOG):
    os.mkdir(TEST_DOG)
print("Make directory ok")

# download & unzip data
URL = "https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip"
urllib.request.urlretrieve(URL, ZIP_DEST)
zip_ref = zipfile.ZipFile(ZIP_DEST, 'r')
zip_ref.extractall(BASE)
zip_ref.close()

# confirm download
print(len(os.listdir(CAT_SOURCE)))
print(len(os.listdir(DOG_SOURCE)))
# Expected Output:
# 12501
# 12501
print("data unzip ok")

# split data into train & test
TRAIN_SIZE = 0.90

def split_data(source, train, test, split):
    # list of image file names
    files = []
    for filename in os.listdir(source):
        file = source / filename
        #print(file)
        if os.path.getsize(file) > 0:
            files.append(filename)
        else:
            print(filename + " is zero length, so ignoring.")

    train_length = int(len(files) * split)
    test_length = int(len(files) - train_length)

    # shuffle dataset images
    shuffled_set = random.sample(files, len(files))

    # define train, test split
    train_set = shuffled_set[0:train_length]
    test_set = shuffled_set[-test_length:]

    # move files to train or test directory
    for filename in train_set:
        this_file = source / filename
        destination = train / filename
        copyfile(this_file, destination)

    for filename in test_set:
        this_file = source / filename
        destination = test / filename
        copyfile(this_file, destination)

# split cat & dog data
split_data(CAT_SOURCE, TRAIN_CAT, TEST_CAT, TRAIN_SIZE)
split_data(DOG_SOURCE, TRAIN_DOG, TEST_DOG, TRAIN_SIZE)

# confirm train/test split
print(len(os.listdir(TRAIN_CAT)),len(os.listdir(TRAIN_DOG)))
print(len(os.listdir(TEST_CAT)), len(os.listdir(TEST_DOG)))
# Expected output: 11250, 11250; 1250, 1250
print("Train test split ok")

# data generators
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
print("ImageDataGenerator ok")

# accuracy callback
ACCURACY_LIMIT = 0.99
class accCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > ACCURACY_LIMIT):
            print(f"Training terminated: reached {ACCURACY_LIMIT*100} accuracy level.")
            self.model.stop_training=True

acc_limit_callback = accCallback()
print("Accuracy callback ok")

# model definition
catdog_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')]
)

catdog_model.summary()
print("Model defintion ok")

# compile model
catdog_model.compile(
    loss='binary_crossentropy',
    optimizer=RMSprop(lr=0.001),
    metrics=['accuracy']
)
print("Model compile ok")


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
test_acc = catdog_history.history['val_accuracy']
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
