#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from keras.utils.np_utils import to_categorical
import tensorflow as tf
import re

try:
    gpu = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpu[0], True)
except:
    print("Couldn't initialize gpu")


# ### Import our data

# In[2]:


imdb = pd.read_csv("datasets/IMDB Dataset.csv")
fina = pd.read_csv("datasets/financial-headlines.csv")
data = pd.concat([imdb, fina])  # combine into big


# ### Define our functions for preprocessing

# In[3]:


def remove_tags(text):
    TAG_RE = re.compile(r"<[^>]+>")
    return TAG_RE.sub("", text)


def preprocess_text(sen):
    # make lowercase
    sentence = sen.lower()

    # Removing html tags
    sentence = remove_tags(sentence)

    # Remove punctuations and numbers
    sentence = re.sub("[^a-zA-Z]", " ", sentence)

    # Single character removal
    # sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)

    # Removing multiple spaces
    sentence = re.sub(r"\s+", " ", sentence)
    return sentence


# ### Process the data
# Remove all neutral scores, capitalization, tags, punctuation, single characters, multiple spaces

# In[4]:


data = data[data["sentiment"] != "neutral"]
data["review"] = [preprocess_text(str(x)) for x in tqdm(data["review"], leave=False)]


# In[5]:


data


# ### Tokenize the data and split X and y

# In[6]:


max_features = 2000
tokenizer = Tokenizer(num_words=max_features, split=" ")
tokenizer.fit_on_texts(data["review"].values)
X = tokenizer.texts_to_sequences(data["review"].values)

Y = pd.get_dummies(data["sentiment"]).values  # convert to indicator columns

test_size = 0.3  # 70/30 train/test split

X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=test_size, random_state=42
)

print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)


# ### Create the model

# In[18]:


embed_dim = 128  # play with these
lstm_out = 196

model = Sequential()
model.add(Embedding(max_features, embed_dim, input_length=X.shape[1]))
model.add(SpatialDropout1D(0.4))
model.add(
    LSTM(lstm_out, dropout=0.2, recurrent_dropout=0.2, recurrent_activation="sigmoid")
)
model.add(Dense(2, activation="softmax"))
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
model.summary()


# ### Fit the model

# In[8]:


batch_size = 32
epochs = 7
model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size, verbose=1)


# ### Test the model

# In[9]:


validation_size = 1500

X_validate = X_test[-validation_size:]
Y_validate = Y_test[-validation_size:]
X_test = X_test[:-validation_size]
Y_test = Y_test[:-validation_size]

score, acc = model.evaluate(X_test, Y_test, verbose=2, batch_size=batch_size)

print(f"score: {score:.2f}")
print(f"accuracy: {acc*100:.2f}%")


# ### Evaluate the model

# In[14]:


pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
for x in trange(len(X_validate)):

    result = model.predict(
        X_validate[x].reshape(1, X_test.shape[1]), batch_size=1, verbose=0
    )[0]

    if np.argmax(result) == np.argmax(Y_validate[x]):
        if np.argmax(Y_validate[x]) == 0:
            neg_correct += 1
        else:
            pos_correct += 1

    if np.argmax(Y_validate[x]) == 0:
        neg_cnt += 1
    else:
        pos_cnt += 1

print("pos acc", pos_correct / pos_cnt * 100, "%")
print("neg acc", neg_correct / neg_cnt * 100, "%")


# In[ ]:
