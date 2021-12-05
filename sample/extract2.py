import pandas as pd
import MeCab
from os import environ
from transformers import BertTokenizer, BertModel
import os.path as osp
import torch
import numpy as np

root_dir = "/autofs/diamond2/share/users/fujie/b3kadai2021/lang"

environ["MECABRC"] = "/etc/mecabrc"
tagger = MeCab.Tagger("-Owakati")

# BERTトークナイザとBERTモデルの準備
model = BertModel.from_pretrained(osp.join(root_dir, 'NICT_BERT-base_JapaneseWikipedia_32K_BPE'))
tokenizer = BertTokenizer.from_pretrained(osp.join(root_dir, 'NICT_BERT-base_JapaneseWikipedia_32K_BPE'))
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)
model.eval()

df = pd.read_csv("data.csv")

def convert_wkt(text):
    """入力文章を分かち書き文に変換
    
    Args:
        text (str): 文章
    
    Returns:
        [str]: 分かち書きされた文 
    
    Examples:
    >>> convert_wkt("吾輩は猫である")
        "吾輩 は 猫　で ある"
    """
    print(text)
    return tagger.parse(text).strip()

def convert_feature(wkt_list):
    """分かち書きされた文章から特徴量を抽出
    
    Args:
        wkt_list (list:str): 分かち書きされた文章のリスト. 次元数=文章数
    
    Returns:
        [ndarray]: CLSトークンの分散表現[文章数, 768]
    """
    #print(wkt_list)
    tokenized = tokenizer(wkt_list, padding=True, truncation=True, return_tensors="pt")
    input_ids = tokenized['input_ids'].to('cuda')

    # 学習しないので誤差逆伝播するための勾配計算モードをオフに
    with torch.no_grad():
        outputs = model(input_ids)
    
    # CLSトークン部分の分散表現
    pooler_output = outputs.pooler_output
    last_hidden_state = outputs.last_hidden_state
    print(last_hidden_state)
    return last_hidden_state.cpu().numpy()

batch_size = 1
feature_list = []
for start in range(0, len(df["text"]), batch_size):
    batch_sentences = df["text"][start: start+batch_size].tolist()
    wakati_list = [convert_wkt(text) for text in batch_sentences]
    print(wakati_list)
    pooler_output = convert_feature(wakati_list)
    print(pooler_output.shape)
    #> ([2, 768])
    #> ([1, 768])
    feature_list.append(pooler_output)

# 変換した特徴量を一つの array にまとめる
feature_array = np.concatenate(feature_list, axis=0)
print(feature_array)
print(feature_array.shape)
