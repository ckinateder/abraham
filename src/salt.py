import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from numpy import array
from sklearn.feature_extraction.text import CountVectorizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, SpatialDropout1D
from sklearn.model_selection import train_test_split
from tqdm import tqdm, trange
from keras.utils.np_utils import to_categorical
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import tensorflow as tf
import re

try:
    gpu = tf.config.experimental.list_physical_devices("GPU")
    tf.config.experimental.set_memory_growth(gpu[0], True)
except:
    print("Couldn't initialize gpu")


class Salty:
    """
    Workflow of using -
    salty = Salty()
    X_train, X_test, Y_train, Y_test, X, Y = salty.prepare_data(data)
    salty.create_model(X)
    salty.fit_model(X_train, Y_train)
    salty.evaluate_model(X_test, Y_test)
    """

    def __init__(
        self, embed_dim=128, lstm_out=196, max_features=2000, epochs=8, batch_size=32
    ):
        self.embed_dim = embed_dim  # play with these
        self.lstm_out = lstm_out
        self.max_features = max_features
        self.epochs = epochs
        self.batch_size = batch_size

    # ### Define our functions for preprocessing
    def remove_tags(self, text):
        TAG_RE = re.compile(r"<[^>]+>")
        return TAG_RE.sub("", text)

    def preprocess_text(self, sen):
        # make lowercase
        sentence = sen.lower()

        # Removing html tags
        sentence = self.remove_tags(sentence)

        # Remove punctuations and numbers
        sentence = re.sub("[^a-zA-Z]", " ", sentence)

        # Single character removal
        # sentence = re.sub(r"\s+[a-zA-Z]\s+", " ", sentence)

        # Removing multiple spaces
        sentence = re.sub(r"\s+", " ", sentence)
        return sentence

    def prepare_data(self, data):
        # Remove all neutral scores, capitalization, tags, punctuation, single characters, multiple spaces
        # takes a dataframe

        data = data[data["sentiment"] != "neutral"]
        print("Data before:")
        print(data)
        data["review"] = [
            self.preprocess_text(str(x))
            for x in tqdm(data["review"], leave=False, desc="process")
        ]
        print("Data after:")
        print(data)
        # ### Tokenize the data and split X and y
        tokenizer = Tokenizer(num_words=self.max_features, split=" ")
        tokenizer.fit_on_texts(data["review"].values)
        X = tokenizer.texts_to_sequences(data["review"].values)
        X = pad_sequences(X)
        Y = pd.get_dummies(data["sentiment"]).values  # convert to indicator columns

        test_size = 0.3  # 70/30 train/test split

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=test_size, random_state=42
        )

        # print(X_train.shape, Y_train.shape)
        # print(X_test.shape, Y_test.shape)
        return X_train, X_test, Y_train, Y_test, X, Y

    # ### Create the model
    def create_model(self, X):
        self.model = Sequential()
        self.model.add(
            Embedding(self.max_features, self.embed_dim, input_length=X.shape[1])
        )
        self.model.add(SpatialDropout1D(0.4))
        self.model.add(
            LSTM(
                self.lstm_out,
                dropout=0.2,
                recurrent_dropout=0.2,
                recurrent_activation="sigmoid",
            )
        )
        self.model.add(Dense(2, activation="softmax"))
        self.model.compile(
            loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
        )
        self.model.summary()
        return self.model

    # ### Fit the model
    def fit_model(self, X_train, Y_train):
        self.model.fit(
            X_train, Y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1
        )
        self.model.save('models/latest-model')
        return self.model

    # ### Test the model
    def evaluate_model(self, X_test, Y_test):
        validation_size = 1500

        X_validate = X_test[-validation_size:]
        Y_validate = Y_test[-validation_size:]
        X_test = X_test[:-validation_size]
        Y_test = Y_test[:-validation_size]

        score, acc = self.model.evaluate(
            X_test, Y_test, verbose=2, batch_size=self.batch_size
        )

        print(f"score: {score:.2f}")
        print(f"accuracy: {acc:.2f}")

        # ### Evaluate the model
        pos_cnt, neg_cnt, pos_correct, neg_correct = 0, 0, 0, 0
        for x in trange(len(X_validate)):
            result = self.model.predict(
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
        pos_acc = pos_correct / pos_cnt * 100
        neg_acc = neg_correct / neg_cnt * 100
        print(f"pos acc: {pos_acc}%\nneg acc:{neg_acc}")
        return {
            "score": score,
            "accuracy": acc,
            "positive accuracy": pos_acc,
            "negative accuracy": neg_acc,
        }


if __name__ == "__main__":
    # ### Import our data
    path = "datasets/"
    imdb = pd.read_csv(path + "IMDB Dataset.csv")
    fina = pd.read_csv(path + "financial-headlines.csv")
    data = pd.concat([imdb, fina])  # combine into big
    salty = Salty()
    X_train, X_test, Y_train, Y_test, X, Y = salty.prepare_data(data)
    salty.create_model(X)
    salty.fit_model(X_train, Y_train)
    salty.evaluate_model(X_test, Y_test)
