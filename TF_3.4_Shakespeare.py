# NLP: Shakespeare
# biderectional LSTM

# import code
import os
import urllib.request
from pathlib import Path
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers
import tensorflow.keras.utils as ku
import numpy as np

# directories & data files
BASE = Path(os.getcwd()) / 'shakespeare'
SONNET_FILE = 'shakespeare/sonnets.txt'
SONNET_URL = 'https://storage.googleapis.com/laurencemoroney-blog.appspot.com/sonnets.txt'
print(f"BASE:    {BASE}")
print(f"SONNET_FILE:    {SONNET_FILE}")
print(f"SONNET_URL:    {SONNET_URL}")
if not os.path.exists(BASE):
    os.mkdir(BASE)
print("\nConfiguration complete\n")

# get data & prep corpus
urllib.request.urlretrieve(SONNET_URL,SONNET_FILE)
print("\nData download complete\n")

# tokenize, make sequences, pad, convert to numpy array
data = open(SONNET_FILE).read()
corpus = data.lower().split("\n")
tokenizer = Tokenizer()
tokenizer.fit_on_texts(corpus)
total_words = len(tokenizer.word_index) + 1
print(f"Total words: {total_words}")
print("\nTokenisation complete\n")

input_seqs = []
for line in corpus:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_seq = token_list[:i+1]
        input_seqs.append(n_gram_seq)

max_seq_len = max([len(x) for x in input_seqs])
input_seqs = np.array(pad_sequences(input_seqs, maxlen=max_seq_len, padding='pre'))
print("\nSequencing complete\n")

# create predictors and label
predictors, label = input_seqs[:,:-1],input_seqs[:,-1]
label = ku.to_categorical(label, num_classes=total_words)    # one hot encoding
print("\nData preprocessing complete\n")

# define model
shakespeare_model = Sequential()
shakespeare_model.add(Embedding(total_words, 100, input_length=max_seq_len-1))
shakespeare_model.add(Bidirectional(LSTM(150, return_sequences = True)))
shakespeare_model.add(Dropout(0.2))
shakespeare_model.add(LSTM(100))
shakespeare_model.add(Dense(total_words/2, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
shakespeare_model.add(Dense(total_words, activation='softmax'))
shakespeare_model.compile(loss='categorical_crossentropy', optimizer=Adam, metrics=['accuracy'])

print(shakespeare_model.summary())
print("\nModel definition complete\n")

# fit model
shakespeare_model_hist = shakespeare_model.fit(predictors, label, epochs=100, verbose=1)
print("\nModel fit complete\n")

# plot learning results
import matplotlib.pyplot as plt
acc = shakespeare_model_hist.history['accuracy']
loss = shakespeare_model_hist.history['loss']
epochs = range(len(acc))

plt.plot(epochs, acc, 'b', label='Training accuracy')
plt.title('Training accuracy')
plt.figure()
plt.plot(epochs, loss, 'b', label='Training Loss')
plt.title('Training loss')
plt.legend()
plt.show()
print("\nPlotting complete\n")

# use model to make a prediction
seed_text = "Help me Obi Wan Kenobi, you're my only hope"
next_words = 100

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=max_seq_len - 1, padding='pre')
    predicted = shakespeare_model.predict_classes(token_list, verbose=0)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
print(seed_text)
print("\nPrediction complete\n")

"""
Help me Obi Wan Kenobi, you're my only hope 
to give another wrong age sight grow cold pain go well bright ' so still thee to thee much did truth did hast seen play 
that worth junes dearer glory prove truth cherish dearer glory down good worth good sum in told thine glory first 
becomes them joy bred confounds thee confounds you live so young to none ill new ill new new new old old heart heart 
on young to call thine eyes sits hence blot for told in joy strong growth of life good woe back confound lie cherish 
go bright days hate bright days twain last thee


WARNING:tensorflow:From /Users/seanmulligan/PycharmProjects/TF_pratice/TF_3.4_Shakespeare.py:95: 
Sequential.predict_classes (from tensorflow.python.keras.engine.sequential) is deprecated and will be removed after 2021-01-01.
Instructions for updating:

Please use instead:

* `np.argmax(model.predict(x), axis=-1)`,   
if your model does multi-class classification   (e.g. if it uses a `softmax` last-layer activation).

* `(model.predict(x) > 0.5).astype("int32")`,   
if your model does binary classification   (e.g. if it uses a `sigmoid` last-layer activation).
"""

