# horse human classification
# transfer learning using pretrained inception model

# import
import tensorflow as tf
from pathlib import Path
import urllib.request
import os
import zipfile
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.optimizers import RMSprop

# define data directories & files
CWD = Path(os.getcwd())
BASE = CWD / 'horsehuman'
TRAIN = BASE / 'train'
TEST = BASE / 'test'
TRAIN_HORSE = TRAIN / 'horses'
TRAIN_HUMAN = TRAIN / 'humans'
TEST_HORSE = TEST / 'horses'
TEST_HUMAN = TEST / 'humans'
print("CWD:        ",CWD)
print("BASE:       ",BASE)
print("TRAIN:      ",TRAIN)
print("TEST:       ",TEST)
print("TRAIN HORSE:",TRAIN_HORSE)
print("TRAIN HUMAN:",TRAIN_HUMAN)
print("TEST HORSE: ",TEST_HORSE)
print("TEST HUMAN: ",TEST_HUMAN)

# make directories
#os.mkdir(BASE)
#os.mkdir(TRAIN)
#os.mkdir(TEST)

# get data: training & testing (validation)
TRAIN_URL = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/horse-or-human.zip"
TEST_URL = "https://storage.googleapis.com/laurencemoroney-blog.appspot.com/validation-horse-or-human.zip"
print("TRAIN URL:  ",TRAIN_URL)
print("TEST URL:   ",TEST_URL)
ZIP_TRAIN = TRAIN / 'horse-or-human.zip'
ZIP_TEST = TEST / 'validation-horse-or-human.zip'
print("TRAIN FILE: ",ZIP_TRAIN)
print("TEST FILE:  ",ZIP_TEST)
"""
urllib.request.urlretrieve(TRAIN_URL,TRAIN / ZIP_TRAIN)
urllib.request.urlretrieve(TEST_URL,TEST / ZIP_TEST)

# unzip data
zip_ref = zipfile.ZipFile(ZIP_TRAIN, 'r')
zip_ref.extractall(TRAIN)
zip_ref.close()
zip_ref = zipfile.ZipFile(ZIP_TEST, 'r')
zip_ref.extractall(TEST)
zip_ref.close()
"""
# confirm data extraction. Expected Output:  500, 527, 128, 128
print(len(os.listdir(TRAIN_HORSE)))
print(len(os.listdir(TRAIN_HUMAN)))
print(len(os.listdir(TEST_HORSE)))
print(len(os.listdir(TEST_HUMAN)))
print("\n data download ok\n")

# data generators with image augmentation
train_image_datagen = ImageDataGenerator(rescale=1./255,
                                          rotation_range=40,
                                          width_shift_range=0.2,
                                          height_shift_range=0.2,
                                          shear_range=0.2,
                                          zoom_range=0.2,
                                          horizontal_flip=True)
train_datagen = train_image_datagen.flow_from_directory(TRAIN,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))

test_image_datagen = ImageDataGenerator(rescale=1./255)     # augmentation not required on test images
test_datagen = test_image_datagen.flow_from_directory(TEST,
                                                    batch_size=20,
                                                    class_mode='binary',
                                                    target_size=(150, 150))
print("\ndatagen ok\n")

# define accuracy callback
ACCURACY_LIMIT = 0.999
class acc_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy')) > ACCURACY_LIMIT:
            print(f"\nTraining terminated: reached {ACCURACY_LIMIT*100}% accuracy.\n")
            self.model.stop_training = True
acc_cb = acc_callback()

print("\ncallback ok\n")

# get inception model weights
INCEP_URL = "https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5"
INCEP_WGTS = BASE / 'incep_v3_weights.h5'
print("INCEP URL:  ",INCEP_URL)
print("INCEP WGTS: ",INCEP_WGTS)

#urllib.request.urlretrieve(INCEP_URL,INCEP_WGTS)
print("\nincep download ok\n")

# instantiate inception model
inception_model = InceptionV3(input_shape = (150, 150, 3),
                                include_top = False,
                                weights = None)
# load weights
WGT = 'horsehuman/incep_v3_weights.h5'
inception_model.load_weights(WGT)

# make layers non-trainable
for layer in inception_model.layers:
  layer.trainable = False

# keep pre-trained weights up to 'mixed7' layer
last_layer = inception_model.get_layer('mixed7')
last_output = last_layer.output

# model summary
inception_model.summary()
print(last_layer.output_shape)
print("\ninception model ok\n")

# add customised final layers to pre-trained inception model
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

horsehuman_model = tf.keras.Model(inception_model.input, x)

horsehuman_model.compile(optimizer=RMSprop(learning_rate=0.0001),
                         loss='binary_crossentropy',
                         metrics=['accuracy'])
# confirm model
horsehuman_model.summary()
print("\ncomplete model ok\n")


# fit model
horsehuman_hist = horsehuman_model.fit(
    train_datagen,
    steps_per_epoch=50,
    epochs=10,
    #validation_steps=50,
    verbose=2,
    callbacks=[acc_cb],
    validation_data=test_datagen
)

print("\nmodel fit ok\n")

# plot accuracy & loss
import matplotlib.pyplot as plt

train_acc = horsehuman_hist.history['accuracy']
test_acc = horsehuman_hist.history['val_accuracy']
train_loss = horsehuman_hist.history['loss']
test_loss = horsehuman_hist.history['val_loss']
num_epochs = range((len(train_acc)))

plt.plot(num_epochs, train_acc, 'r', label='Training Accuracy')
plt.plot(num_epochs, test_acc, 'b', label='Testing Accuracy')
plt.title("Horse Human Accuracy")
plt.legend(loc=0)
plt.figure()

plt.plot(num_epochs, train_loss, 'r', label='Training Loss')
plt.plot(num_epochs, test_loss, 'b', label='Testing Loss')
plt.title("Horse Human Loss")
plt.legend(loc=0)
plt.figure()
print("\nmodel results ok\n")
plt.show()

