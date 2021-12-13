#ファイルをimport
import torch
torch.set_default_tensor_type('torch.cuda.FloatTensor')
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import _pickle as cPickle
# import cPickle
import tqdm
#bertように追加
import pandas as pd
import MeCab
from os import environ
from transformers import BertTokenizer, BertModel
import os.path as osp
import numpy as np
root_dir = "/autofs/diamond2/share/users/fujie/b3kadai2021/lang"

environ["MECABRC"] = "/etc/mecabrc"
tagger = MeCab.Tagger("-Owakati")

# BERTトークナイザとBERTモデルの準備
bmodel = BertModel.from_pretrained(osp.join(root_dir, 'NICT_BERT-base_JapaneseWikipedia_32K_BPE'))
tokenizer = BertTokenizer.from_pretrained(osp.join(root_dir, 'NICT_BERT-base_JapaneseWikipedia_32K_BPE'))
bdevice = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
bmodel.to(bdevice)
bmodel.eval()

def convert_wkt(text):
    return tagger.parse(text).strip()

def convert_feature(wkt_list):
    tokenized = tokenizer(wkt_list, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized['input_ids'].to('cuda')
    with torch.no_grad():
        outputs = bmodel(input_ids)
    pooler_output = outputs.pooler_output
    last_hidden_state = outputs.last_hidden_state
    return last_hiddn_state
#GPUの使用を指定
device = 'cpu'
# device = 'cpu'

#再現性を持たせるため乱数を固定
torch.manual_seed(1)

# def prepare_sequence(seq, to_ix):
#     idxs = [to_ix[w] for w in seq]
#     tensor = torch.LongTensor(idxs)
#     return autograd.Variable(tensor)

#系列データをテンソル化
def prepare_sequence(idxs):
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

#データの前処理
# target_dir = '/mnt/d/data/tagger/dataset/'
target_dir = './'
#tag_to_ix
#sents_idx
word_to_ix,tag_to_ix,sents_idx,labels_idx = cPickle.load(open(target_dir + "kakikomi3.pkl", "rb"))

# training_data = [
#     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
#     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# ]
# word_to_ix = {}
# for sent, tags in training_data:
#     for word in sent:
#         if word not in word_to_ix:
#             word_to_ix[word] = len(word_to_ix)
# print(word_to_ix)
# tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

print(word_to_ix['I'])
#print(word_to_ix[0])
print(labels_idx[0])
#print(sents_idx[0])
# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 64
HIDDEN_DIM = 32

#LSTMの処理
class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        #self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True) #双方向に変更
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        # self.hidden = self.init_hidden()
        self.hidden = None

    def init_hidden(self):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly
        # why they have this dimensionality.
        # The axes semantics are (num_layers, minibatch_size, hidden_dim)
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def forward(self, sentence):
        wakati_list = convert_wkt(sentence)
        embeds = convert_feature(wakati_list)
        file = open('embeds.csv','a')
        file.write(str(embeds))
        #file.write('\n')
        #file.write("next")
        file.write('\n')
        file.close()
        # lstm_out, self.hidden = self.lstm(
        #     embeds.view(len(sentence), 1, -1), self.hidden)
        lstm_out, _ = self.lstm(
            embeds.view(len(sentence), 1, -1), None)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
model = model.to(device)

# import ipdb; ipdb.set_trace()
# model = model.cuda()

# inputs = prepare_sequence(training_data[0][0], word_to_ix)
#inputs = prepare_sequence(sents_idx[0])
inputs = prepare_sequence(labels_idx[0])
inputs = inputs.to(device)
tag_scores = model(inputs)
_, pred_tag = torch.max(tag_scores.data, 1)
print(pred_tag)

loss_function = nn.NLLLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1)
def optimizer_to(optim, device):
        for param in optim.state.values():
            # Not sure there are any global tensors in the state dict
            if isinstance(param, torch.Tensor):
                param.data = param.data.to(device)
                if param._grad is not None:
                    param._grad.data = param._grad.data.to(device)
            elif isinstance(param, dict):
                for subparam in param.values():
                    if isinstance(subparam, torch.Tensor):
                        subparam.data = subparam.data.to(device)
                        if subparam._grad is not None:
                            subparam._grad.data = subparam._grad.data.to(device)

optimizer = optim.RMSprop(model.parameters())
optimizer_to(optimizer, device)
EPOCHS = 20
count = 1
for epoch in range(EPOCHS):  # again, normally you would NOT do 300 epochs, it is toy data
#     for sentence, tags in training_data:

    print("epoch",  epoch + 1 , "/" , EPOCHS)
    
    total_loss = 0

    for i, (sentence, tags) in tqdm.tqdm(enumerate(zip(sents_idx, labels_idx))):
        # print(i)
        # import ipdb; ipdb.set_trace()
        
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Also, we need to clear out the hidden state of the LSTM,
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Variables of word indices.
        sentence_in = prepare_sequence(sentence)
        targets = prepare_sequence(tags)

        sentence_in = sentence_in.to(device)
        targets = targets.to(device)

        batch_size = 1
        feature_list = []
        # Step 3. Run our forward pass.
        with open("bdata.txt") as f:
            line = f.readlines()[count]
            sentence = line
        count += 1
        #wakati_list = convert_wkt(line)
        #sentence_in = wakati_list
        #sentence_in = convert_feature(wakati_list)
        
        tag_scores = model(sentence)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

        # import ipdb; ipdb.set_trace()
        
        # total_loss += loss.data[0]
        total_loss += float(loss.detach().cpu().numpy())
        
    print('loss: %.4f' % loss)
plus = 0
# inputs = prepare_sequence(training_data[0][0], word_to_ix)
word_to_ix,tag_to_ix,sents_idx,labels_idx = cPickle.load(open(target_dir + "rekakikomi.pkl", "rb"))
for i in range(10):
    inputs = prepare_sequence(sents_idx[plus])
    tag_scores = model(inputs)
    _, pred_tag = torch.max(tag_scores.data, 1)
    plus += 1
    print(pred_tag)

