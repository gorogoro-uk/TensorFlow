# NLP: GLOVE transfer learning
# LSTM

# import code
import tensorflow as tf
import os
from pathlib import Path
import urllib.request
import csv
import random
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt

# constants
embed_dim = 100
max_length = 16
trunc_type = 'post'
pad_type = 'post'
oov_tok = "<OOV>"
train_size = 160000
test_ratio = 0.10

# data directories
CWD = Path(os.getcwd())
BASE = CWD / 'glove'
if not os.path.exists(BASE):
    os.mkdir(BASE)
DATA_FILE = BASE / 'training_cleaned.csv'
DATA_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/training_cleaned.csv'
print(f"CWD:        {CWD}")
print(f"BASE:       {BASE}")
print(f"DATA_FILE:  {DATA_FILE}")
print(f"DATA_URL:   {DATA_URL}")
print("\nConfig ok ok\n")

# get data file
urllib.request.urlretrieve(DATA_URL, DATA_FILE)
print("\nData download ok\n")

# read data csv
num_sent = 0
corpus = []
with open(DATA_FILE) as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        row_item = []
        row_item.append(row[5])
        row_label = row[0]
        if row_label == '0':
            row_item.append(0)
        else:
            row_item.append(1)
        num_sent = num_sent + 1
        corpus.append(row_item)

print(num_sent, len(corpus))
print(corpus[1])
# Expected Output: 1600000, 1600000
# ["is upset that he can't update his Facebook by texting it... and might cry as a result  School today also. Blah!", 0]
print("\nRead data csv ok\n")

# shuffle data & split into data & labels
random.shuffle(corpus)
sentences=[]
labels=[]
for x in range(train_size):
    sentences.append(corpus[x][0])
    labels.append(corpus[x][1])

# tokenize data
train_tokenizer = Tokenizer()
train_tokenizer.fit_on_texts(sentences)
word_index = train_tokenizer.word_index
vocab_size = len(word_index)
train_seqs = train_tokenizer.texts_to_sequences(sentences)
train_pad_seqs = pad_sequences(train_seqs, maxlen=max_length, padding=pad_type, truncating=trunc_type)
print("\nTokenize ok\n")

# split inot train / test & convert to numpy
split = int(test_ratio * train_size)
train_data = np.array(train_pad_seqs[split:train_size])
train_labels = np.array(labels[split:train_size])
test_data = np.array(train_pad_seqs[0:split])
test_labels = np.array(labels[0:split])

print(vocab_size)
print(word_index['i'])
# Expected Output: 138858, 1
print("\nTrain - test split  ok\n")


# get word embeddings: 100 dimension version of GloVe from Stanford
GLOVE_FILE = BASE / 'glove.6B.100d.txt'
GLOVE_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/glove.6B.100d.txt'
print(f"GLOVE_FILE: {GLOVE_FILE}")
print(f"GLOVE_URL:  {GLOVE_URL}")
urllib.request.urlretrieve(GLOVE_URL, GLOVE_FILE)
print("\nGLOVE download ok\n")

# read embeddings file
embed_index = {}
with open(GLOVE_FILE) as f:
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embed_index[word] = coefs

# get word embeddings for our vocab list
embed_matrix = np.zeros((vocab_size+1, embed_dim))
for word, i in word_index.items():
    embed_vector = embed_index.get(word)
    if embed_vector is not None:
        embed_matrix[i] = embed_vector

print(len(embed_matrix))
# Expected Output: 138859
print("\nGlove data prep ok\n")

# define model
glove_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size+1, embed_dim, input_length=max_length, weights=[embed_matrix], trainable=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Conv1D(64, 5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=4),
    tf.keras.layers.LSTM(64),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
glove_model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
glove_model.summary()

# fit model
glove_model_hist = glove_model.fit(train_data, train_labels,
                    epochs=50,
                    validation_data=(test_data, test_labels),
                    verbose=2)
print("\nTraining Complete\n")


# plot learning
acc = glove_model_hist.history['accuracy']
val_acc = glove_model_hist.history['val_accuracy']
loss = glove_model_hist.history['loss']
val_loss = glove_model_hist.history['val_loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'r')
plt.plot(epochs, val_acc, 'b')
plt.title('Training and validation accuracy')
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend(["Accuracy", "Validation Accuracy"])
plt.figure()

plt.plot(epochs, loss, 'r')
plt.plot(epochs, val_loss, 'b')
plt.title('Training and validation loss')
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend(["Loss", "Validation Loss"])
plt.figure()

