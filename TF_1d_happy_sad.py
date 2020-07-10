# Happy-Sad image classification

import tensorflow as tf
import wget
import urllib.request
import os
#from pathlib import path
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop

# data import
# wget does not work with python (but does in Google colab)
#!wget --no-check-certificate \
#    "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip" \
#    -O "/hs/happy-or-sad.zip"

# create directory in root
#os.mkdir(os.path.join(os.getcwd(), 'hs'))
#os.mkdir(os.path.join(os.getcwd(), 'hs/happy_sad'))

zip_dest = os.path.join(os.getcwd(),'/hs/happy-or-sad.zip')
unzip_dest = os.path.join(os.getcwd(),'/hs/happy_sad/')
print(zip_dest, unzip_dest)


zip_url = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/happy-or-sad.zip"
urllib.request.urlretrieve(zip_url, zip_dest)

zip_ref = zipfile.ZipFile(zip_dest, 'r')
zip_ref.extractall(unzip_dest)
zip_ref.close()

# data generator
image_data_gen = ImageDataGenerator(rescale=1/255.0)

train_data_gen = image_data_gen.flow_from_directory(
    "/hs/happy_sad",
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
happysad_model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D(2,2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
happysad_model.model_summary()

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
