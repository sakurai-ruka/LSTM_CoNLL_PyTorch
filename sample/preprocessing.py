#!/usr/bin/env python
# coding: utf-8

# In[99]:


import codecs
import collections
from torchtext import data, datasets
import pickle as cPickle

target_dir = '/home/sakurai/git/LSTM_CoNLL_PyTorch/sample/'
train_file = target_dir + 'kakikomi.txt'


# In[95]:


def read_corpus(lines):
    """
    convert corpus into features and labels
    """
    features = list()
    labels = list()
    tmp_fl = list()
    tmp_ll = list()
    for line in lines:
        if not (line.isspace() or (len(line) > 10 and line[0:10] == '-DOCSTART-')):
            line = line.rstrip('\n').split()
            tmp_fl.append(line[0])
            tmp_ll.append(line[-1])
        elif len(tmp_fl) > 0:
            features.append(tmp_fl)
            labels.append(tmp_ll)
            tmp_fl = list()
            tmp_ll = list()
    if len(tmp_fl) > 0:
        features.append(tmp_fl)
        labels.append(tmp_ll)

    return features, labels


# In[96]:


def make_dic(doc_sent, doc_labels):
    word2idx = collections.defaultdict(int)
    label2idx = collections.defaultdict(int)
    for sent, labels in zip(doc_sent, doc_labels):
        for word, label in zip(sent, labels):
            if word not in word2idx:
                word2idx[word] = len(word2idx)
            if label not in label2idx: 
                label2idx[label] = len(label2idx)
        word2idx[u'<unk>'] = len(word2idx)
    return word2idx, label2idx


# In[97]:


with codecs.open(train_file, 'r', 'utf-8') as f:
    lines = f.readlines()

# converting format
train_features, train_labels = read_corpus(lines)
# make dictionary
word2idx, label2idx = make_dic(doc_sent=train_features, doc_labels=train_labels)


# In[98]:


sents_idx = [[word2idx[word] for word in sent] for sent in train_features]
labels_idx = [[label2idx[label] for label in labels] for labels in train_labels]


# In[101]:


cPickle.dump([word2idx, label2idx, sents_idx, labels_idx], open(target_dir + "kakikomi.pkl", "wb"))

