# Happy-Sad image classification

import tensorflow as tf
from pathlib import Path
# import wget
import urllib.request
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# data import
#!wget --no-check-certificate \
#    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
#    -O "/hs/happy-or-sad.zip"

# define locations
BASE = Path(os.getcwd()) / 'happysad'
HS_DATA = BASE / 'hs_data'
ZIP_DEST = BASE / 'happy-or-sad.zip'
ZIP_URL = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip"
print(f"BASE:     {BASE}")
print(f"HS_DATA:  {HS_DATA}")
print(f"ZIP_DEST: {ZIP_DEST}")
print(f"ZIP_URL:  {ZIP_URL}")

# make directories
if not os.path.exists(BASE):
    os.mkdir(BASE)
if not os.path.exists(HS_DATA):
    os.mkdir(HS_DATA)

# download data file & unzip
urllib.request.urlretrieve(ZIP_URL, ZIP_DEST)
zip_ref = zipfile.ZipFile(ZIP_DEST, 'r')
zip_ref.extractall(HS_DATA)
zip_ref.close()

# data generator
image_data_gen = ImageDataGenerator(rescale=1/255.0)

train_data_gen = image_data_gen.flow_from_directory(
    HS_DATA,
    target_size=(150, 150),
    batch_size=10,
    class_mode='binary'
)

# accuracy callback
ACCURACY_LIMIT = 0.995
class accCallBack(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') > ACCURACY_LIMIT):
            print("\nTraining terminated: reached desired accuracy {ACCURACY_LIMIT*100}%\n")
            self.model.stop_training = True

happysad_acc_callback = accCallBack()

# model definition
# input: [batch=10, height=150, width=150, rgb=3]
happysad_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),    # output: [10, 148, 148, 16]
    tf.keras.layers.MaxPooling2D(2,2),                                                  # output: [10, 74, 74, 16]
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),                              # output: [10, 72, 72, 32]
    tf.keras.layers.MaxPooling2D(2, 2),                                                 # output: [10, 36, 36, 32]
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),                              # output: [10, 34, 34, 32]
    tf.keras.layers.MaxPooling2D(2, 2),                                                 # output: [10, 17, 17, 32]
    tf.keras.layers.Flatten(),                                                          # output: [10, 9248]
    tf.keras.layers.Dense(512, activation='relu'),                                      # output: [10, 512]
    tf.keras.layers.Dense(1, activation='sigmoid')                                      # output: [10, 1]
])
happysad_model.summary()

# compile model
happysad_model.compile(
    optimizer=RMSprop(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# fit model
happysad_model.fit(
    train_data_gen,
    steps_per_epoch=8,
    epochs=10,
    verbose=1,
    callbacks=[happysad_acc_callback]
)
