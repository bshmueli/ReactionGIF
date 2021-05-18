# -*- coding: utf-8 -*-

import os, random
import torch
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, accuracy_score, precision_score, recall_score

# reproducibility
seed = 42
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

file = 'ReactionGIF.json'
df = pd.read_json(file, lines=True)
train_data, test_data = train_test_split(df, random_state=43, test_size=0.1)
train_data = train_data.copy()
test_data = test_data.copy()
train_data['label'] = pd.Categorical(train_data['label'])
train_data['labels'] = train_data['label'].cat.codes
categories = train_data['label'].cat.categories

test_data['label'] = pd.Categorical(test_data['label'], categories=categories)
test_data['labels'] = test_data['label'].cat.codes
def report(gold, pred):
    print(classification_report(gold, pred, digits=3))
    acc = accuracy_score(gold, pred)
    p = precision_score(gold, pred, average='weighted')
    r = recall_score(gold, pred, average='weighted')
    f1 = f1_score(gold, pred, average='weighted')
    print(f'{acc*100:.1f} & {p*100:.1f} & {r*100:.1f} & {f1*100:.1f}')

"""#Majority"""

majority = train_data['labels'].value_counts().keys()[0]
test_data['pred'] = majority
report(test_data['labels'], test_data['pred'])

"""#TFIDF"""

from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=1000, stop_words='english') # using default parameters
vectorizer.fit(train_data['text'])
train_X = vectorizer.transform(train_data['text'])
test_X = vectorizer.transform(test_data['text'])
print(train_X.shape, test_X.shape)

"""#LogisticRegressionCV

"""

from sklearn.linear_model import LogisticRegressionCV


model = LogisticRegressionCV(Cs=3, cv=5, verbose=1, max_iter=1000, n_jobs=-1)
model.fit(train_X, train_data['labels'])
pred_y = model.predict(test_X)
report(test_data['labels'], pred_y)

"""#RoBERTa"""

# !pip install tqdm
# !pip install transformers
# !pip install simpletransformers
# !pip install tensorboardx

from simpletransformers.classification import ClassificationModel
model = ClassificationModel('roberta', 'roberta-base',
                            num_labels=len(categories))

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
report(test_data['labels'], test_data['pred'])
