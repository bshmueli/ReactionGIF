# ReactionGIF code
This repository contains the code used for the baselines in the paper [Happy Dance, Slow Clap: Using Reaction GIFs to Predict Induced Affect on Twitter](https://arxiv.org/abs/2105.09967)

Included are the following Python files:
- ``sentiment.py``: code for majority, LR, and RoBERTa for **sentiment binary classification**
- ``reactions.py``: code for majority, LR, and RoBERTa for **reaction multi-class classification**
- ``emotions.py``: code for majority, LR, and RoBERTa for **emotions multi-label classification**
- ``CNN.py``: CNN code for **sentiment**, **reaction**, and **emotions**  classification

In addition, the reaction to emotions mappings are available in:
- ``Reactions2GoEmotions.csv``
