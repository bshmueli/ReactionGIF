# -*- coding: utf-8 -*-

import random, os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score


# reproducibility
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

positive = [
    'hug',
    'kiss',
    'wink',
    'awww',
    'hearts',
    'win',
    'fist_bump',
    'high_five',
    'good_luck',
    'you_got_this',
    'ok',
    'thumbs_up',
    'agree',
    'yes',
    'dance',
    'happy_dance',
    'applause',
    'slow_clap',
    'popcorn',
    'thank_you',
    ]

file = 'ReactionGIF.json'
df = pd.read_json(file, lines=True)
train_data, test_data = train_test_split(df, random_state=43, test_size=0.1)
train_data = train_data.copy()
test_data = test_data.copy()
train_data['labels'] = train_data['label'].copy().apply(lambda x: 1 if x in positive else 0)
test_data['labels'] = test_data['label'].copy().apply(lambda x: 1 if x in positive else 0)


def report(gold, pred):
    print(classification_report(gold, pred, digits=3))
    acc = accuracy_score(gold, pred,  )
    p = precision_score(gold, pred, average='weighted')
    r = recall_score(gold, pred, average='weighted')
    f1 = f1_score(gold, pred, average='weighted')
    print(f'{acc*100:.1f} & {p*100:.1f} & {r*100:.1f} & {f1*100:.1f}')

"""#Majority"""

majority = train_data['labels'].value_counts().keys()[0]
test_data['pred'] = majority
report(test_data['labels'], test_data['pred'])

"""#TFIDF/LR"""

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=1000, stop_words='english') # using default parameters
vectorizer.fit(train_data['text'])
train_X = vectorizer.transform(train_data['text'])
test_X = vectorizer.transform(test_data['text'])
print(train_X.shape, test_X.shape)

model = LogisticRegressionCV(verbose=1, dual=False, max_iter=1000)
model.fit(train_X, train_data['labels'])
pred_y = model.predict(test_X)
report(test_data['labels'], pred_y)

"""#RoBERTa"""

# !pip install tqdm
# !pip install transformers
# !pip install simpletransformers
# !pip install tensorboardx

from simpletransformers.classification import ClassificationModel
model = ClassificationModel('roberta', 'roberta-base')
train_data.drop(columns=['label'], inplace=True)
test_data.drop(columns=['label'], inplace=True)
model.train_model(train_data, args={
    'fp16': False,
    'overwrite_output_dir': True,
    'train_batch_size': 32,
    "eval_batch_size": 32,
    'max_seq_length': 96,
    'num_train_epochs': 3
})
_, model_outputs, _ = model.eval_model(test_data)
test_data['probabilities'] = model_outputs.tolist()
test_data['pred'] = test_data['probabilities'].apply(lambda x: x.index(max(x)))
print(classification_report(test_data['labels'], test_data['pred'], digits=3))
report(test_data['labels'], test_data['pred'])
