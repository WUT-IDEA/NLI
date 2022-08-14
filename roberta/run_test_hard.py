import csv
import pandas as pd                         #导入pandas包
import re
import datetime
import math
import numpy as np
import heapq
import json
def create_df(column_name):
    # df = df + file_name
    df = pd.DataFrame(columns=column_name)
#     print('df',df)
    return df
def append_df(df,column_name, column_value):
    thisdict = dict.fromkeys(column_name)
    for i in range(len(column_name)):
#         print('column_name[i]',column_name[i])
#         print('column_value[i]',column_value[i])
        thisdict[column_name[i]] = column_value[i]
    add_data = pd.Series(thisdict)
#     print('add_data', add_data)
    # ignore_index=True不能少
    df = df.append(add_data, ignore_index=True)
#     print('df',df)
    return df
def write_file(df,out_dir, file_name):
    df.to_csv(out_dir + file_name+'.csv',mode='a',encoding = 'utf-8',header=None, index=False,sep = ',')

# # # 创建dataframe
column_name = ('predicted_label', 'true_label','is_equal', 'layer1', 'layer2','layer3','layer4','layer5','layer6','layer7','layer8','layer9','layer10','layer11','layer12','max',
               'label0_Probability','label1_Probability','label2_Probability')
df = create_df(column_name)
file_name = 'hardsnli_score_predicted'
out_dir = '/data1/home/zmj/roberta/trained_model/snli_lr1e5_maxlen128_wd1e2_3/epoch4step20000/'
df.to_csv(out_dir+file_name+'.csv',encoding = 'utf-8',header=True, index=False,sep = ',')

# 超参数
learning_rate = 1e-5
weight_decay = 1e-2
hidden_dropout_prob = 0.1
epochs = 6
batch_size = 16
max_len = 128 #并不是越长越好，样本数据集的长度最长是78
model_name ="/data1/home/zmj/roberta/trained_model/snli_lr1e5_maxlen128_wd1e2_3/epoch4step20000/"
num_labels = 6
output_hidden_states=True
output_attentions=True
gradient_accumulation_steps = 1
warmup_steps=100
max_steps = 1
adam_epsilon = 1e-8

from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, AdamW
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase

# 定义device
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")

@dataclass
class DataCollatorForCLS:
    tokenizer: PreTrainedTokenizerBase


    def __call__(
        self, examples: List[Union[List[int], torch.Tensor, Dict[str, torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        
        # examples 是一个字典的list，利用以下代码使list转换成tensor
        if isinstance(examples[0], (dict, BatchEncoding)):
            batch = self.tokenizer.pad(examples, return_tensors="pt")
        else:
            batch = {"input_ids": _collate_batch(examples, self.tokenizer)}

        # special_tokens_mask在这里被删掉
#         special_tokens_mask = batch.pop("special_tokens_mask", None)
        
        return batch


# 定义tokenizer方法
def tokenize_function(examples):
    return tokenizer(
        examples['textp'], # textp编码
        examples['texth'], # texth编码
        padding="max_length",
        truncation=True,
        max_length=max_len
#         return_special_tokens_mask=True
    )



def test(model, iterator, criterion, device):
    now_time = datetime.datetime.now()
    print(now_time, "【======test start======】")
    model.eval()
    epoch_loss = 0
    epoch_acc = 0
    total_num = 0

    with torch.no_grad():
        for _, batch in enumerate(iterator):
            i=1
            labels = batch["label"]
            attention_mask = batch["attention_mask"]
            input_ids = batch["input_ids"]

            # label改变shape
            labels = labels.unsqueeze(1)

            # 需要 LongTensor
            input_ids, labels, attention_mask = input_ids.long(), labels.long(), attention_mask.long()

            # 梯度清零
            optimizer.zero_grad()

            # 迁移到GPU
            input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)

            # 获取模型输出
            output = model(
                input_ids=input_ids, 
                labels=labels, 
                attention_mask=attention_mask
            )
#             loss = output[0]
            y_pred_prob = output[1]
#             y_pred_label = y_pred_prob.argmax(dim=1)
            y_pred_label = y_pred_prob.argmax(dim=2)
            
            f = nn.Softmax(dim = 2)
            y_pred_prob_softmax = f(y_pred_prob)
#             print('y_pred_prob_softmax:',y_pred_prob_softmax)

            #score
            score = output[2]
#             print('score.shape', score.shape)

            now_time = datetime.datetime.now()
            print(now_time, '===========START: 将', y_pred_prob.size(0)*i, 'score和predicted写入'+ out_dir + file_name +'=========')
            # print('score.size(0):',score.size(0))
            for i in range(score.size(0)):
#                 print(score[i][0])
                column_value = []
                predicted_label = y_pred_label[i][0].item()
                true_label = labels[i][0].item()
                column_value.append(predicted_label)
                column_value.append(true_label)
                column_value.append(predicted_label == true_label)

                for j in range(12):
                    each = score[i][0][j].item()
                    each = format(each, '.4f')
                    column_value.append(each)

                max = (score[i][0].argmax(axis=0)+1).item()
                max= format(max, '.0f')
                column_value.append(max)

                for k in range(3):
                    each = y_pred_prob_softmax[i][0][k].item()
                    each = format(each, '.4f')
                    column_value.append(each)
                    
            #     print('max:',max)
            #     print('layers:',layers)
                df = create_df(column_name)
                df = append_df(df,column_name, column_value)
                write_file(df,out_dir,file_name)

            now_time = datetime.datetime.now()
            print(now_time, '===========END:将',y_pred_prob.size(0)*i, 'score和predicted写入'+ out_dir + file_name +'=========')

            # 这个 loss 和 output[0] 是一样的
            loss = criterion(y_pred_prob.view(-1, num_labels), labels.view(-1))

            # 计算loss和acc
#             acc = ((y_pred_label == labels.view(-1)).sum()).item()
            acc = ((y_pred_label == labels).sum()).item()
#             print('y_pred_label:',y_pred_label)
#             print('y_pred_label.size(0):',y_pred_label.size(0))

#             now_time = datetime.datetime.now()
#             print(now_time, '===========START: 将predicted result写入'+ out_dir + file_name_pre +'=========')
#             for i in range(y_pred_label.size(0)):
#                 column_value_pre = []
#                 predicted_label = y_pred_label[i][0].item()
#                 true_label = labels[i][0].item()
#                 column_value_pre.append(predicted_label)
#                 column_value_pre.append(true_label)
#                 column_value_pre.append(predicted_label == true_label)

#                 df_pre= create_df(column_name_pre)
#                 df_pre = append_df(df_pre,column_name_pre, column_value_pre)
#                 write_file(df_pre,out_dir,file_name)

#             now_time = datetime.datetime.now()
#             print(now_time, '===========END: 将predicted result写入'+ out_dir + file_name_pre +'=========')



            epoch_loss += loss.item()
            epoch_acc += acc
            total_num += y_pred_prob.size(0)
            i += 1
#     now_time = datetime.datetime.now()
#     print(now_time, '===========END: 将predicted result写入'+ out_dir + file_name +'=========')
    print('epoch_acc',epoch_acc)
    print('total_num',total_num)
#     print('score.shape', score.shape)
    now_time = datetime.datetime.now()
    print(now_time, "【======test end======】")
    return epoch_loss / len(iterator), epoch_acc / total_num



# 加载PLM和tokenizer
config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_labels,output_hidden_states=output_hidden_states, output_attentions=output_attentions, hidden_dropout_prob=hidden_dropout_prob)
model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", padding=True, truncation=True)
model.to(device)

# 加载数据
data_dir = "/data1/home/zmj/roberta/dataset/snli/"
data_files = {"train":data_dir+"snlitrain.csv", "dev":data_dir+"snlidev.csv", "test":data_dir+"hard_test.csv"}
raw_datasets = load_dataset("csv", data_files=data_files, sep=",")

# 编码
tokenized_datasets = raw_datasets.map(
                tokenize_function,
                batched=True,
                num_proc=2, # 多个进程一起编码
                remove_columns=['textp','texth'],
                desc="Running tokenizer on dataset line_by_line",
            )

dc = DataCollatorForCLS(tokenizer=tokenizer)

# 使用dataloader加载数据
params = {"batch_size": batch_size, "shuffle": True, "num_workers": 2}
test_dataloader = DataLoader(tokenized_datasets['test'], collate_fn=dc, **params)

# 定义优化器
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)

# 定义损失函数
criterion = nn.CrossEntropyLoss()

# # 测试结果
   
print("【======测试开始======】")

# 测试集看效果
test_loss, test_acc= test(model, test_dataloader, criterion, device)
print("test loss: ", test_loss, "\t", "test acc:", test_acc)
    