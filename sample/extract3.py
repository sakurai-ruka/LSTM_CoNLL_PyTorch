import pandas as pd
import MeCab
from os import environ
from transformers import BertTokenizer, BertModel
import os.path as osp
import torch
import numpy as np
import csv

root_dir = "/autofs/diamond2/share/users/fujie/b3kadai2021/lang"

environ["MECABRC"] = "/etc/mecabrc"
tagger = MeCab.Tagger("-Owakati")

# BERTトークナイザとBERTモデルの準備
model = BertModel.from_pretrained(osp.join(root_dir, 'NICT_BERT-base_JapaneseWikipedia_32K_BPE'))
tokenizer = BertTokenizer.from_pretrained(osp.join(root_dir, 'NICT_BERT-base_JapaneseWikipedia_32K_BPE'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

file = open("bdata.txt",'r')
def convert_wkt(text):
    return tagger.parse(text).strip()

def convert_feature(wkt_list):
    tokenized = tokenizer(wkt_list, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized['input_ids'].to('cuda')

    with torch.no_grad():
        outputs = model(input_ids)
    pooler_output = outputs.pooler_output
    last_hidden_state = outputs.last_hidden_state
    file = open('bkakikomi.csv','a')
    #last_hidden_state = last_hidden_state.replace('\n','')
    file.write(str(last_hidden_state))
    #file.write('\n')
    #file.write('next')
    file.write('\n')
    file.close
    #print(last_hidden_state)
    return last_hidden_state.cpu().numpy()

batch_size = 1
feature_list = []
file1 = open("bdata.txt",'r')
for line in file1:
    print(line)
    wakati_list =convert_wkt(line)
    #batch_sentences = ["text"][start: start+batch_size].tolist()
    #wakati_list = [convert_wkt(text) for text in batch_sentences]
    #print(wakati_list)
    pooler_output = convert_feature(wakati_list)
file1.close()
