# test new sentence

# import library
import torch
import gluonnlp as nlp
import numpy as np
import datetime as dt
from kobert import get_tokenizer
from kobert import get_pytorch_kobert_model
from torch.utils.data import Dataset, DataLoader
from bert_dataset import BERTDataset
from bert_classifier import BERTClassifier

# GPU
device = torch.device("cuda:3")

# vocab
bertmodel, vocab = get_pytorch_kobert_model(cachedir=".cache")

# Setting parametrs
max_len = 64
batch_size = 64

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)

def predict(predict_sentence):
    test_eval = "dysphoria"
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model1 = BERTClassifier(bertmodel, dr_rate=0.5).to(device)
    model1.load_state_dict(torch.load("./trained_model/2022-10-23/71.7_2022_10_23_21_39_34_best_trained_model_okt.pt"))
    # 71.7_2022_10_23_21_39_34_best_trained_model_okt
    #71.5_2022_10_23_21_55_13_best_trained_model_okt
    model1.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model1(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()

            if np.argmax(logits) == 0:
                test_eval = "euphoria"
            elif np.argmax(logits) == 1:
                test_eval = "dysphoria"
            elif np.argmax(logits) == 2:
                test_eval = "neutral"

    return test_eval

import pandas as pd
public_test_df = pd.read_csv("../KERC22Dataset/public_test_data.tsv", sep="\t")
private_test_df = pd.read_csv("../KERC22Dataset/private_test_data.tsv", sep="\t")


private_test_df['context'].fillna("", inplace=True)
public_test_df['context'].fillna("", inplace=True)

# train dataset for KoBERT
public_and_private_test_list = []
for sentence_id, context, sentence in zip(public_test_df['sentence_id'],public_test_df['context'], public_test_df['sentence']):
    data = []
    data.append(str(sentence_id))
    # print(sentence+context)
    data.append(sentence+context)

    public_and_private_test_list.append(data)
    
for sentence_id, context, sentence in zip(private_test_df['sentence_id'],private_test_df['context'], private_test_df['sentence']):
    data = []
    data.append(str(sentence_id))
    # print(sentence+context)
    data.append(sentence+context)

    public_and_private_test_list.append(data)

now_date = dt.datetime.now()

id_list = []
predicted_label_list = []

cnt = 0;
for i in range(0, len(public_and_private_test_list)):
    id_list.append(public_and_private_test_list[i][0])
    predicted_label_list.append(predict(public_and_private_test_list[i][1]))
    if i%10 == 0:
        print('current i = {}'.format(i))

result_df = pd.DataFrame({"Id": id_list, "Predicted": predicted_label_list})
print(result_df.sample(10))

# make folder about datetime
import os

path = './result_{}'.format(now_date.strftime("%Y-%m-%d"))
if not os.path.isdir(path):
    os.mkdir(path)
    
result_df.to_csv(path + '/result_{}.csv'.format(now_date.strftime("%H_%M_%S")), index=False)
