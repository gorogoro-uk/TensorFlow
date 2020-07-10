# NLP: BBC News
# word embedding vectors

from pathlib import Path
import os
import csv
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# prepare directories
BBC = Path(os.getcwd()) / 'bbc'
BBC_FILE = BBC / 'bbc-text.csv'
print(f"BBC:        {BBC}")
print(f"BBC_FILE:   {BBC_FILE}")

# parameters
vocab_size = 1000
embedding_dim = 16
max_length = 120
trunc_type='post'
padding_type='post'
oov_tok = "<OOV>"
train_ratio = 0.8

# stopword list
stopwords = [ "a", "about", "above", "after", "again", "against", "all", "am", "an", "and", "any", "are", "as", "at",
              "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "could", "did",
              "do", "does", "doing", "down", "during", "each", "few", "for", "from", "further", "had", "has", "have",
              "having", "he", "he'd", "he'll", "he's", "her", "here", "here's", "hers", "herself", "him", "himself",
              "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "it", "it's", "its",
              "itself", "let's", "me", "more", "most", "my", "myself", "nor", "of", "on", "once", "only", "or", "other",
              "ought", "our", "ours", "ourselves", "out", "over", "own", "same", "she", "she'd", "she'll", "she's",
              "should", "so", "some", "such", "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
              "then", "there", "there's", "these", "they", "they'd", "they'll", "they're", "they've", "this", "those",
              "through", "to", "too", "under", "until", "up", "very", "was", "we", "we'd", "we'll", "we're", "we've",
              "were", "what", "what's", "when", "when's", "where", "where's", "which", "while", "who", "who's", "whom",
              "why", "why's", "with", "would", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
              "yourself", "yourselves" ]
print(len(stopwords))
# Expected Output: 153
print("\nDirectories, parameters, stoplist ok\n")

# prepare datasets: remove stopwords
sentences = []
labels = []
with open(BBC_FILE, 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        sentence = row[1]
        for word in stopwords:
            token = " " + word + " "
            sentence = sentence.replace(token, " ")
            sentence = sentence.replace("  ", " ")
        sentences.append(sentence)

# split into train & test
train_size = int(len(sentences) * train_ratio)
train_sentences = sentences[:train_size]
test_sentences = sentences[train_size:]
train_labels = labels[:train_size]
test_labels = labels[train_size:]

print(train_size)
print(len(train_sentences))
print(len(train_labels))
print(len(test_sentences))
print(len(test_labels))
print("\nSplit datasets ok\n")
# Expected output (if training_portion=0.8): 1780, 1780, 1780, 445, 445

# tokenize train data
tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_sentences)
word_index = tokenizer.word_index
train_seq = tokenizer.texts_to_sequences(train_sentences)
train_pad = pad_sequences(train_seq, padding=padding_type, maxlen=max_length)

print(len(train_seq[0]),len(train_pad[0]))
print(len(train_seq[1]),len(train_pad[1]))
print(len(train_seq[10]),len(train_pad[10]))
print("\nTokenize train data ok\n")
# Expected Ouput: 449, 120, 200, 120, 192, 120

# tokenize test data
test_seq = tokenizer.texts_to_sequences(test_sentences)
test_pad = pad_sequences(test_seq, padding=padding_type, maxlen=max_length)

print(len(test_seq),test_pad.shape)
print("\nTokenize test data ok\n")
# Expected output: 445, (445, 120)

# tokenize labels
label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

# convert to Numpy arrays
train_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
test_label_seq = np.array(label_tokenizer.texts_to_sequences(test_labels))

print(train_label_seq[0],train_label_seq[1],train_label_seq[2])
print(train_label_seq.shape)
print(test_label_seq[0],test_label_seq[1],test_label_seq[2])
print(test_label_seq.shape)
print("\nTokenize labels ok\n")
# Expected output: [4], [2], [1], (1780, 1), [5], [4], [3], (445, 1)

# define model
bbc_model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, input_length=max_length),
    tf.keras.layers.GlobalAveragePooling1D(),
    tf.keras.layers.Dense(24, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')
])
bbc_model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
print("\nModel ok\n")
bbc_model.summary()
# Expected Output
# Layer (type)                 Output Shape              Param #
# =================================================================
# embedding (Embedding)        (None, 120, 16)           16000
# _________________________________________________________________
# global_average_pooling1d (Gl (None, 16)                0
# _________________________________________________________________
# dense (Dense)                (None, 24)                408
# _________________________________________________________________
# dense_1 (Dense)              (None, 6)                 150
# =================================================================
# Total params: 16,558
# Trainable params: 16,558
# Non-trainable params: 0

# Expected output: (1000, 16)

# fit model
num_epochs = 30
bbc_model_hist = bbc_model.fit(train_pad, train_label_seq,
                    epochs=num_epochs,
                    validation_data=(test_pad, test_label_seq),
                    verbose=2)
print("\nTraining ok\n")

# plot training results
import matplotlib.pyplot as plt

def plot_graphs(hist, string):
    plt.plot(hist.history[string])
    plt.plot(hist.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(bbc_model_hist, "accuracy")
plot_graphs(bbc_model_hist, "loss")
print("\nPlots ok\n")



"""
# visualise word embedding vectors for our project

# get embedding vectors from model
e = bbc_model.layers[0]
weights = e.get_weights()[0]
print(weights.shape)

# reverse word index look up
#reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

#def decode_sentence(text):
#   return ' '.join([reverse_word_index.get(i, '?') for i in text])

# visualise word vectors
import io

out_v = io.open('vecs.tsv', 'w', encoding='utf-8')
out_m = io.open('meta.tsv', 'w', encoding='utf-8')
for word_num in range(1, vocab_size):
  word = reverse_word_index[word_num]
  embeddings = weights[word_num]
  out_m.write(word + "\n")
  out_v.write('\t'.join([str(x) for x in embeddings]) + "\n")
out_v.close()
out_m.close()

try:
  from google.colab import files
except ImportError:
  pass
else:
  files.download('vecs.tsv')
  files.download('meta.tsv')
"""
