import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Dense, MaxPooling1D
from keras.layers import Conv1D, Embedding, GlobalMaxPooling1D
from keras.models import Model
from sklearn.metrics import roc_auc_score

MAX_SEQUENCE_LENGTH = 100
MAX_VOCAB_SIZE = 20000
EMBEDDING_DIM = 100 ## gloVe comes in specific sizes
VALIDATION_SPLIT = 0.2
BATCH_SIZE = 128
EPOCHS = 10

word2vec = {}
with open('../../../passport/toxic-comments/glove.6B/glove.6B.100d.txt') as file:
    for line in file:
        values = line.split()
        word = values[0] # first item is word
        vec  = np.asarray(values[1:], dtype='float32') #rest of line is emb
        word2vec[word] = vec
#print(len(word2vec)) #40k 
#print(word2vec['and'])

training = pd.read_csv(#path to comments.csv)
comments = train["comment_text"].values
labs = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
targs = training[labs].values
