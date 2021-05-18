# -*- coding: utf-8 -*-

# !wget 'https://reactiongif.imfast.io/glove.twitter.27B.100d.txt.pickle'
# !pip install emoji

# !wget http://nlp.stanford.edu/data/glove.twitter.27B.zip
# !unzip glove.twitter.27B.zip

import numpy as np
import pandas as pd
import tensorflow as tf

from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, label_ranking_average_precision_score
from sklearn.preprocessing import OneHotEncoder
import os, pickle, re, random
from nltk.tokenize import TweetTokenizer
from keras import optimizers
from keras.layers import Input, Embedding, LSTM, Dense, Flatten, GlobalMaxPooling1D, GlobalMaxPool1D, Dropout, Convolution1D, Bidirectional, Concatenate, GlobalAveragePooling1D
from keras.layers.core import Activation, Dropout, Dense
from keras.layers.embeddings import Embedding
from keras.models import Model, load_model, Sequential
from keras.optimizers import RMSprop, Adam
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

 # reproducibility
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

"""# Lib"""

class BasicPreprocess:
  def __init__(self, min_tokens, max_length, emoji=True):
    self.min = min_tokens
    self.max = max_length
    self.emoji = emoji
    self.tknzr = TweetTokenizer()

  def _massage(self, s):
    if self.emoji:
      s = emoji.demojize(s)
      s = s.replace('_', ' ')
    s = s.lower()
    s = re.sub(r'@[a-z0-9_]+', '@user', s)
    s = re.sub(r'^(@user )+', '', s)
    s = re.sub(r'( @user)+$', '', s)
    s = ' '.join(self.tknzr.tokenize(s)[0:self.max])
    return s

  def _filter(self, s):
    if 'http' in s:
      return False
    if len(s.split()) < self.min:
      return False
    return True

  def _massage(self, s):
    s = s.lower()
    s = re.sub(r'@[a-z0-9_]+', '@user', s)
    s = re.sub(r'^(@user )+', '', s)
    s = re.sub(r'( @user)+$', '', s)
    s = ' '.join(self.tknzr.tokenize(s)[0:self.max])
    return s

class Glove:
  def __init__(self, file):
    self.file = file

  def loadPickledGlove(self):
    print('Loading Pickled Glove')
    embeddings_index = dict()
    with open(self.file, 'rb') as pickle_file:
      embeddings_index = pickle.load(pickle_file)
    return embeddings_index

  def loadGloveModel(self):
    print("Loading Glove Model")
    f = open(self.file,'r')
    embeddings_index = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        wordEmbedding = np.array([float(value) for value in splitLines[1:]])
        embeddings_index[word] = wordEmbedding
    print(len(embeddings_index)," words loaded!")
    return embeddings_index

class Embed:
  def __init__(self, pretrained, max_length):
    glove = Glove(pretrained)
    self.embeddings = glove.loadGloveModel()
    self.tknzr = TweetTokenizer()
    self.max_length = max_length

  def tokenize(self, documents):
    return [self.tknzr.tokenize(document) for document in documents]

  def fit(self, documents):
    documents = self.tokenize(documents)
    words = [item for sublist in documents for item in sublist]
    print('Vocab: {}'.format(len(words)))
    freqs = Counter(words)
    words = [word for (word, count) in freqs.items() if count > 1]
    words.insert(0, 'UNK')
    words.insert(0, 'PAD')
    self.word2id = {word:index for index, word in enumerate(words)}
    self.id2word = {index:word for index, word in enumerate(words)}

  def documents_to_sequences(self, documents):
    documents = self.tokenize(documents)
    unk = self.word2id.get('UNK')
    documents = [[self.word2id.get(token, unk) for token in document[0:self.max_length]] for document in documents]

    for i,v in enumerate(documents):
      if len(v) > 40:
        print(i, v, len(v))
        exit()
    documents = [document[0:self.max_length] + [self.word2id['PAD']] * (self.max_length - len(document)) for document in documents]
    print(documents[0:2])
    for i,v in enumerate(documents):
      if len(v) != 40:
        print(i, len(v))
        exit()
    documents = np.array(documents)
    return documents

  def restore(self, sequence):
    return ' '.join([self.id2word[i] for i in sequence if self.id2word[i] != 'PAD'])

  def matrix(self):
    dims = len(self.embeddings['the'])
    embedding_matrix = np.random.random((len(self.word2id), dims))
    not_found = 0
    for word, i in self.word2id.items():
      vector = self.embeddings.get(word)
      if vector is not None:
        embedding_matrix[i] = vector
      else:
        not_found += 1
    print("Embedding completed ({} words not found)".format(not_found))
    return embedding_matrix

"""# Get Data"""

embed = Embed('glove.twitter.27B.100d.txt', 40)
preprocess = BasicPreprocess(0, 80, emoji=True)


def get_data(task):
    file = 'ReactionGIF.json'
    df = pd.read_json(file, lines=True)
    train_data, test_data = train_test_split(df, random_state=42, test_size=0.1)
    train_data = train_data.copy()
    test_data = test_data.copy()

    if task == 'reaction':
        train_data['label'] = pd.Categorical(train_data['label'])
        train_data['labels'] = train_data['label'].cat.codes
        categories = train_data['label'].cat.categories

        test_data['label'] = pd.Categorical(test_data['label'], categories=categories)
        test_data['labels'] = test_data['label'].cat.codes

        train_y = to_categorical(train_data['labels'])
        test_y = to_categorical(test_data['labels'])

    if task == 'emotion':
        train_data['label'] = pd.Categorical(train_data['label'])
        categories = train_data['label'].cat.categories
        test_data['label'] = pd.Categorical(test_data['label'], categories=categories)

        reactions2emotions = pd.read_csv('Reactions2GoEmotions.csv', index_col='Reaction').astype('int')
        emotions = reactions2emotions.columns

        for emotion in reactions2emotions.columns:
            if len(reactions2emotions[emotion].value_counts()) == 1:
                print('Dropping ', emotion)
                reactions2emotions.drop(columns=[emotion], inplace=True)
        emotions = reactions2emotions.columns

        train_data = train_data.join(reactions2emotions, on='label')
        train_data['labels'] = train_data[emotions].values.tolist()
        print(train_data)
        test_data = test_data.join(reactions2emotions, on='label')
        test_data['labels'] = test_data[emotions].values.tolist()

    if task == 'sentiment':
        positive = ['hug', 'kiss', 'wink', 'awww', 'hearts', 'win', 'fist_bump', 'high_five',
                    'good_luck', 'you_got_this', 'ok', 'thumbs_up', 'agree', 'yes', 'dance',
                    'happy_dance', 'applause', 'slow_clap', 'popcorn', 'thank_you']

        train_data['labels'] = train_data['label'].copy().apply(lambda x: 1 if x in positive else 0)
        test_data['labels'] = test_data['label'].copy().apply(lambda x: 1 if x in positive else 0)

    train_data['text'] = train_data['text'].apply(lambda x: preprocess._massage(x))
    test_data['text'] = test_data['text'].apply(lambda x: preprocess._massage(x))
    text = train_data['text']
    text.append(test_data['text'])
    embed.fit(text)

    X_train = embed.documents_to_sequences(train_data['text'])
    X_test = embed.documents_to_sequences(test_data['text'])

    embedding_matrix = embed.matrix()
    if task == 'emotion':
        y_train = np.array(train_data[emotions].values.tolist())
        y_test = np.array(test_data[emotions].values.tolist())
    else:
        y_train = to_categorical(train_data['labels'])
        y_test = test_data['labels']
    return X_train, y_train, embedding_matrix, X_test, y_test

def report(task, gold, pred):
    if task == 'emotion':
        print(f'LRAP {label_ranking_average_precision_score(y_test, y_pred):.3f}')
        print(f'{label_ranking_average_precision_score(y_test, y_pred):.3f}')
    else:
        acc = accuracy_score(gold, pred,  )
        p = precision_score(gold, pred, average='weighted')
        r = recall_score(gold, pred, average='weighted')
        f1 = f1_score(gold, pred, average='weighted')
        print(f'Accuracy {acc*100:.1f}, Precision {p*100:.1f}, Recall {r*100:.1f}, F1 {f1*100:.1f}')
        print(f'{acc*100:.1f} & {p*100:.1f} & {r*100:.1f} & {f1*100:.1f}')

def cnn(X_train, y_train, embedding_matrix, X_test, y_test):
    MAX_SEQUENCE_LENGTH = 40
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedding_layer = Embedding(embedding_matrix.shape[0],
                                embedding_matrix.shape[1],
                                weights = [embedding_matrix],
                                input_length = MAX_SEQUENCE_LENGTH,
                                trainable=True,
                                name = 'embeddings')(sequence_input)
    units = 100
    dropout = 0.2
    epochs = 100
    n_classes = y_train.shape[1]
    print(n_classes)
    batch_size = 128
    learning_rate = 0.0005
    filename = 'model.hd5'

    x = Convolution1D(filters=units,
                        kernel_size=3,
                        padding="valid",
                        activation="relu",
                        strides=1)(embedding_layer)

    x = GlobalMaxPool1D()(x)
    x = Dense(units, activation="relu")(x)
    x = Dropout(dropout)(x)
    x = Dense(units, activation="relu")(x)
    x = Dropout(dropout)(x)
    preds = Dense(n_classes, activation="softmax")(x)
    model = Model(sequence_input, preds)
    init_weights = model.get_weights()

    opt = Adam(learning_rate=learning_rate)
    if n_classes in [2,26]: # no neutral, no grief
        loss = 'binary_crossentropy'
    else:
        loss='categorical_crossentropy'
    model.compile(loss=loss,
                optimizer=opt,
                metrics=['accuracy'])

    print('Fitting')
    model.set_weights(init_weights)
    earlyStopping = EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
    mcp_save = ModelCheckpoint(filename, save_best_only=True, monitor='val_loss', mode='min', verbose=0)
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=1, min_delta=1e-5, mode='min')
    model.fit(X_train, y_train, epochs = epochs, verbose=1,
            callbacks=[earlyStopping, mcp_save, reduce_lr_loss], batch_size=batch_size, validation_split=0.2)

    print('Predicting')
    model = load_model(filename)
    y_pred = model.predict(X_test, batch_size=batch_size)
    return y_pred

for task in ['reaction', 'sentiment', 'emotion']:
    X_train, y_train, embedding_matrix, X_test, y_test = get_data(task)
    y_pred = cnn(X_train, y_train, embedding_matrix, X_test, y_test)
    print(task)
    if task in ['reaction', 'sentiment']:
        y_pred = np.argmax(y_pred,axis=1)
    report(task, y_test, y_pred)
