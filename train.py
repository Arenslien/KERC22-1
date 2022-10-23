# ########################
# 1-1. KoBERT 기본 환경 설정
# ########################

# import torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import numpy as np
import gluonnlp as nlp
import datetime as dt
import os

# import kobert
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model

# import transformers
from transformers import AdamW
from transformers.optimization import get_cosine_schedule_with_warmup

# ########################
# 1-2. GPU Setting
# ########################
# GPU Setting
print("[GPU Setting]")
print("- cuda.is_available's value =", torch.cuda.is_available())
device = torch.device("cuda:3")
print("- device =", device,"\n")

# BERT Model 불러오기
bertmodel, vocab = get_pytorch_kobert_model(cachedir="cache")
# print(bertmodel)

# ########################
# 2. 데이터셋 가져오기
# ########################

# Data 가져오기
import pandas as pd

# train data 가져오기
train_data_df = pd.read_csv("../KERC22Dataset/train_data.tsv", sep="\t")
print("[Train data's length]")
print("- train_data length =", len(train_data_df), "\n")
print("[Train data sample]")
print(train_data_df.sample(1), "\n")

# train data label 가져오기
train_data_labels_df = pd.read_csv('../KERC22Dataset/train_labels.csv')
print("[Train label sample]")
print(train_data_labels_df.sample(1), "\n")

# NA 값 채우기
train_data_df['context'].fillna("", inplace=True)

# label 변경 euphoria, dysphoria or neutral --> 0, 1, 2
train_data_labels_df.loc[(train_data_labels_df['label'] == "euphoria"), 'label'] = 0
train_data_labels_df.loc[(train_data_labels_df['label'] == "dysphoria"), 'label'] = 1
train_data_labels_df.loc[(train_data_labels_df['label'] == "neutral"), 'label'] = 2

print("[Changed Label]")
# train_data
print(train_data_labels_df.sample(1), "\n")

# ########################
# 3. 데이터셋 분리(train : validation = 6 : 1)
# ########################

# Train & Validation Data List
train_data_list = []
validation_data_list = []

# Train Data Scene별로 그룹핑 --> Scene별로 dictionary
train_data_dict = dict()
for sentence, scene, context, label in zip(train_data_df['sentence'], train_data_df['scene'], train_data_df['context'], train_data_labels_df['label']):
    data = []
    data.append(sentence+context)
    data.append(label)

    if scene not in train_data_dict:
        train_data_dict[scene] = [data]
    else:
        train_data_dict[scene].append(data)

# scene으로 그룹핑된 데이터들 담기
train_data_scenes = []
for key, val in train_data_dict.items():
    train_data_scenes.append(val)

# train_data_scenes 셔플하기
import random
random.shuffle(train_data_scenes)

# train & test 6:1 비율로 나누기
threshold = int(len(train_data_df)/7*6)

cnt = 0
for scenes in train_data_scenes:
    if cnt <= threshold:
        for data in scenes:
            train_data_list.append(data)
    else:
        for data in scenes:
            validation_data_list.append(data)
    cnt = cnt + len(scenes)

# 비율로 잘 나누어졌는지 확인
print("[Train & Validation Set's Length]")
print("- train length : {}".format(len(train_data_list)))
print("- validation length : {}".format(len(validation_data_list)))
print("- sum : {}".format(len(train_data_list)+len(validation_data_list)),"\n")

# ########################
# 3-2. 분리된 데이터 토큰화
# ########################

# 토크나이저 가져오기
from konlpy.tag import Okt
from konlpy.tag import Komoran
from konlpy.tag import Hannanum
from konlpy.tag import Kkma
okt = Okt()
komoran = Komoran()
hannanum = Hannanum()
kkma = Kkma()

okt_stop_tags = ['Josa', 'Eomi', 'KoreanParticle', 'Number', 'Punctuation']
okt_tags = ['Adjective', 'Noun', 'Verb']

# Okt
def self_tokenizer(string):
    tokenized_string = ''
    pos_result = okt.pos(string)

    for data in pos_result:
        if data[1] not in okt_stop_tags:
            tokenized_string = tokenized_string + data[0] + ' '

    return tokenized_string

# tags = ['JK', 'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ', 'JX', 'JC', 'EP', 'EF', 'EC', 'ETN', 'ETM']

for i in range(len(train_data_list)):
    train_data_list[i][0] = self_tokenizer(train_data_list[i][0])

for i in range(len(validation_data_list)):
    validation_data_list[i][0] = self_tokenizer(validation_data_list[i][0])

# ########################
# 4. KoBERT 입력 데이터로 만들기
# ########################

# KoBERT dataset
from bert_dataset import BERTDataset

# Setting parameters
max_len = 64 # 처음 32 안좋음

warmup_ratio = 0.1
num_epochs = 20
max_grad_norm = 1
log_interval = 200
batch_size = 32
learning_rate =  45e-6
dr_rate = 0.4
print("[Parameters]")
print("batch_size : {}".format(batch_size))
print("learning_rate : {}".format(learning_rate))
print("dr_rate : {}".format(dr_rate))


#토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

train_data = BERTDataset(train_data_list, 0, 1, tok, max_len, True, False)
validation_data = BERTDataset(validation_data_list, 0, 1, tok, max_len, True, False)

# torch 형식의 dataset 만들기
train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=5,shuffle=True)
validation_dataloader = torch.utils.data.DataLoader(validation_data, batch_size=batch_size, num_workers=5)

# ########################
# 5. KoBERT 학습모델 만들기
# ########################
from bert_classifier import BERTClassifier

#BERT 모델 불러오기
model = BERTClassifier(bertmodel,  dr_rate=dr_rate).to(device) # 변경 예정

#optimizer와 schedule 설정
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]

optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
loss_fn = nn.CrossEntropyLoss()

t_total = len(train_dataloader) * num_epochs
warmup_step = int(t_total * warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_step, num_training_steps=t_total)

#정확도 측정을 위한 함수 정의
def calc_accuracy(X,Y):
    max_vals, max_indices = torch.max(X, 1)
    train_acc = (max_indices == Y).sum().data.cpu().numpy()/max_indices.size()[0]
    return train_acc

print(train_dataloader)


# ########################
# 6. KoBERT 모델 학습시키기
# ########################

# load model
#model.load_state_dict(torch.load("./trained_model/trained_model.pt"))
best_acc=0
for e in range(num_epochs):
    train_acc = 0.0
    test_acc = 0.0
    model.train()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(train_dataloader):
        optimizer.zero_grad()
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        loss = loss_fn(out, label)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        optimizer.step()
        scheduler.step()  # Update learning rate schedule
        train_acc += calc_accuracy(out, label)
        if batch_id % log_interval == 0:
            print("epoch {} batch id {} loss {} train acc {}".format(e+1, batch_id+1, loss.data.cpu().numpy(), train_acc / (batch_id+1)))
    print("epoch {} train acc {}".format(e+1, train_acc / (batch_id+1)))

    model.eval()
    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(validation_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)
        valid_length= valid_length
        label = label.long().to(device)
        out = model(token_ids, valid_length, segment_ids)
        test_acc += calc_accuracy(out, label)
    print("epoch {} validation acc {}".format(e+1, test_acc / (batch_id+1)))

    # Model 저장을 위한 디렉토리 준비
    now_date = dt.datetime.now()
    path = "/{}".format(now_date.strftime("%Y-%m-%d"))
    if not os.path.isdir('./trained_model' + path):
        os.mkdir("./trained_model" + path)

    # save model
    if((test_acc/(batch_id+1))>best_acc):
        torch.save(model.state_dict(), "./trained_model" + path + "/{}_{}_{}_best_trained_model_okt.pt".format(round(test_acc/(batch_id+1), 3)*100, now_date.strftime("%Y_%m_%d"), now_date.strftime("%H_%M_%S")))
        best_acc = test_acc/(batch_id+1)
        print("Saved best model")

    if e % 10 == 0:
        torch.save(model.state_dict(), "./trained_model" + path + "/{}_{}_trained_model_okt.pt".format(round(test_acc/(batch_id+1), 3)*100, now_date.strftime("%H_%M_%S")))
        print("Saved model")
