# -*- coding: utf-8 -*-

import os, random
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, label_ranking_average_precision_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
import torch

# reproducibility
seed = 0
os.environ['PYTHONHASHSEED'] = str(seed)
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

file = 'ReactionGIF.json'
df = pd.read_json(file, lines=True)
train_data, test_data = train_test_split(df, random_state=43, test_size=0.1)
train_data = train_data.copy()
train_data['label'] = pd.Categorical(train_data['label'])

categories = train_data['label'].cat.categories

test_data = test_data.copy()

test_data['label'] = pd.Categorical(test_data['label'], categories=categories)

reactions2emotions = pd.read_csv('Reactions2GoEmotions.csv', index_col='Reaction').astype('int')

for emotion in reactions2emotions.columns:
    if (len(reactions2emotions[emotion].value_counts()) == 1) or (emotion == 'Neutral'):
        print('Dropping ', emotion)
        reactions2emotions.drop(columns=[emotion], inplace=True)
emotions = reactions2emotions.columns

train_data = train_data.join(reactions2emotions, on='label')
train_data['labels'] = train_data[emotions].values.tolist()

test_data = test_data.join(reactions2emotions, on='label')
test_data['labels'] = test_data[emotions].values.tolist()

def perf(gold, pred):
    print(label_ranking_average_precision_score(gold, pred))

"""#Majority"""

majority = train_data[emotions].sum()
majority /= sum(majority)
test_data['pred'] = len(test_data) * [majority.values.tolist()]

perf(test_data['labels'].tolist(), test_data['pred'].tolist())

"""#TFIDF"""

from sklearn.linear_model import LogisticRegressionCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import StratifiedKFold, KFold
from skmultilearn.problem_transform import BinaryRelevance

vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=1000, stop_words='english') # using default parameters
vectorizer.fit(train_data['text'])
train_X = vectorizer.transform(train_data['text'])
test_X = vectorizer.transform(test_data['text'])
print(train_X.shape, test_X.shape)

classifier = BinaryRelevance(LogisticRegressionCV(Cs=3, cv=StratifiedKFold(n_splits=5), verbose=1, max_iter=1000, n_jobs=-1))
print(np.array(train_data['labels'].tolist()))
print(classifier)

classifier.fit(train_X, np.array(train_data['labels'].tolist()))

predictions = classifier.predict_proba(test_X)

perf(test_data['labels'].tolist(), predictions.toarray())

"""#RoBERTa"""
# !pip install tqdm
# !pip install transformers
# !pip install simpletransformers
# !pip install tensorboardx
#

from simpletransformers.classification import ClassificationModel, MultiLabelClassificationModel
model = MultiLabelClassificationModel('roberta', 'roberta-base',
                                        num_labels=len(emotions))

model.train_model(train_data, args={
    'fp16': False,
    'overwrite_output_dir': True,
    'train_batch_size': 32,
    "eval_batch_size": 32,
    'max_seq_length': 96,
    'num_train_epochs': 3
})

metrics, model_outputs, _ = model.eval_model(test_data)
print(metrics['LRAP'])
perf(test_data['labels'].tolist(), model_outputs.tolist())
