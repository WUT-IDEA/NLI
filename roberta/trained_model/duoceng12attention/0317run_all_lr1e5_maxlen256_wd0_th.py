# 超参数
learning_rate = 1e-5
weight_decay = 0
hidden_dropout_prob = 0.1
epochs = 6
batch_size = 16
max_len = 256 #并不是越长越好，样本数据集的长度最长是78
model_name ="roberta-base"
num_labels = 3
output_hidden_states=True
output_attentions=True



import datetime
from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, AdamW,get_linear_schedule_with_warmup
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from torch.utils.tensorboard import SummaryWriter #是一个将数据发送到tensorboard的类

# 定义device
device = torch.device("cuda: 1" if torch.cuda.is_available() else "cpu")
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

#             # 梯度清零
#             optimizer.zero_grad()

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
        
#             #score
#             score = output[2]
# #             print('score.shape', score.shape)



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
    now_time = datetime.datetime.now()
#     print(now_time, '===========END: 将predicted result写入'+ out_dir + file_name_pre +'=========')
    print(now_time, '===========test结果=========')
    print('epoch_acc',epoch_acc)
    print('total_num',total_num)
    return epoch_loss / len(iterator), epoch_acc, epoch_acc / total_num

# 定义训练方法
def train(model, iterator, test_dataloader, hardtest_dataloader, easytest_dataloader, device, num_labels, criterion, epoch_num):
    # 定义优化器
    lr_record = []
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = 0.8)
    total_steps = len(iterator) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)

    print(datetime.datetime.now(), "【======train start======】")
    model.train()
    epoch_loss = 0
    epoch_acc = 0
    total_num = 0
    step = 0
    stepresult_list = []
    for i, batch in enumerate(iterator):
        
        labels = batch["label"]
        attention_mask = batch["attention_mask"]
        input_ids = batch["input_ids"]
        
        
        step +=1
        
        # 标签形状为 (batch_size, 1)
        labels = labels.unsqueeze(1)
        
        # 需要 LongTensor
        input_ids, labels, attention_mask = input_ids.long(), labels.long(), attention_mask.long()
        
        # 梯度清零
        optimizer.zero_grad()
        
        # 迁移到GPU
        input_ids, labels, attention_mask = input_ids.to(device), labels.to(device), attention_mask.to(device)
#         print("========before==========")
#         print(labels)
#         print(attention_mask)
#         print(input_ids)
        # 获取模型输出
        output = model(
            input_ids=input_ids, 
            labels=labels, 
            attention_mask=attention_mask
        )
#         loss = output[0]
        y_pred_prob = output[1]
    
#         print('y_pred_prob', y_pred_prob)
#         print('y_pred_prob.shape', y_pred_prob.shape)
#         print('y_pred_label', y_pred_prob.argmax(dim=2))
#         y_pred_label = y_pred_prob.argmax(dim=1)
        y_pred_label = y_pred_prob.argmax(dim=2)
#         print('y_pred_label', y_pred_label)
#         print('y_pred_label.shape', y_pred_label.shape) 
# #         [16,1]
#         print('labels:', labels)
#         print('labels.shape', labels.shape)
        
#         #score
#         score = output[2]
# #         print('score.shape', score.shape)
        
        # 这个 loss 和 output[0] 是一样的
        loss = criterion(y_pred_prob.view(-1, num_labels), labels.view(-1))
        
        # 反向传播
        loss.backward()
        optimizer.step()
        scheduler.step()
#         lr_record.append(scheduler.get_lr()[0])
        
        
        # 计算acc
#         print('acc', y_pred_label == labels.view(-1))
#         print('labels.view(-1)',labels.view(-1))
#         print('(y_pred_label == labels.view(-1)).sum()',(y_pred_label == labels.view(-1)).sum())
#         print('(y_pred_label == labels.view).sum()',(y_pred_label == labels).sum())
#         print('labels',labels)
#         acc = ((y_pred_label == labels.view(-1)).sum()).item()
        acc = ((y_pred_label == labels).sum()).item()
#         print('y_pred_label:', y_pred_label)
#         print('labels:', labels)
#         print('acc', acc)
        
        # epoch 中的 loss 累加
        epoch_loss += loss.item()
        epoch_acc += acc
        total_num += y_pred_prob.size(0)
        
#         print('epoch_acc',epoch_acc)
#         print('total_num',total_num)
#         print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(labels)))
    #         print(('(len(iterator) * iterator.batch_size)', len(iterator) * iterator.batch_size))
#         print(' len(iterator)',  len(iterator))
#         print('iterator.batch_size',iterator.batch_size)
        if i % 100 == 0:
            print(datetime.datetime.now(), i)
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(labels)))
    
        if i % 10000 == 0:
            print(datetime.datetime.now(), i)
            print("current loss:", epoch_loss / (i+1), "\t", "current acc:", epoch_acc / ((i+1)*len(labels)))
            
            # 验证集看效果
            test_loss, test_correct, test_acc= test(model, test_dataloader, criterion, device)
            print("test loss: ", test_loss, "\t", "test acc:", test_acc, "\t", "test_correct:", test_correct)
            now_time = datetime.datetime.now()
            print(now_time, "【======test end======】")
            hardtest_loss, hardtest_correct, hardtest_acc= test(model, hardtest_dataloader, criterion, device)
            print("hardtest loss: ", hardtest_loss, "\t", "hardtest acc:", hardtest_acc, "\t", "hardtest_correct:", hardtest_correct)
            now_time = datetime.datetime.now()
            print(now_time, "【======test end======】")
            easytest_loss, easytest_correct, easytest_acc= test(model, easytest_dataloader, criterion, device)
            print("easytest loss: ", easytest_loss, "\t", "easytest acc:", easytest_acc, "\t", "easytest_correct:", easytest_correct)
            now_time = datetime.datetime.now()
            print(now_time, "【======test end======】")
            stepresult_list.append(i)
            stepresult_list.append('test result')
            stepresult_list.append(test_loss)
            stepresult_list.append(test_acc)
            stepresult_list.append(test_correct)
            stepresult_list.append('hard test result')
            stepresult_list.append(hardtest_loss)
            stepresult_list.append(hardtest_acc)
            stepresult_list.append(hardtest_correct)
            stepresult_list.append('easy test result')
            stepresult_list.append(easytest_loss)
            stepresult_list.append(easytest_acc)
            stepresult_list.append(easytest_correct)
            # 保存模型
            print(datetime.datetime.now(),i, "【======保存"+"epoch"+str(epoch_num)+"step" + str(i) + "的模型======】")
            output_dir = '/data1/home/zmj/roberta/trained_model/duoceng12attention/trained_model/0317snli_lr1e5_maxlen128_wd0_th/epoch'+str(epoch_num)+'step'+str(i)
            model.save_pretrained(output_dir)
    print('epoch_acc',epoch_acc)
    print('total_num',total_num)
    print(datetime.datetime.now(), "【======train end======】")
    return epoch_loss / len(iterator), epoch_acc, epoch_acc / total_num, stepresult_list



# 加载PLM和tokenizer
config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_labels,output_hidden_states=output_hidden_states, output_attentions=output_attentions, hidden_dropout_prob=hidden_dropout_prob)
model = RobertaForSequenceClassification.from_pretrained(model_name, config=config)
tokenizer = RobertaTokenizer.from_pretrained("roberta-base", padding=True, truncation=True)
model.to(device)

# 加载数据
data_dir = "/data1/home/zmj/roberta/dataset/snli/"
# data_files = {"train":data_dir+"snlitrain21.csv", "test":data_dir+"snlitest21.csv", "hardtest":data_dir+"snlitest21.csv", "easytest":data_dir+"snlitest21.csv"}
data_files = {"train":data_dir+"snlitrain.csv", "test":data_dir+"snlitest.csv", "hardtest":data_dir+"hard_test.csv", "easytest":data_dir+"easy_test.csv"}
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
params_test = {"batch_size": batch_size, "shuffle": False, "num_workers": 2}
train_dataloader = DataLoader(tokenized_datasets['train'], collate_fn=dc, **params)
test_dataloader = DataLoader(tokenized_datasets['test'], collate_fn=dc, **params_test)
hardtest_dataloader = DataLoader(tokenized_datasets['hardtest'], collate_fn=dc, **params_test)
easytest_dataloader = DataLoader(tokenized_datasets['easytest'], collate_fn=dc, **params_test)

# # 定义优化器
# # lr_list = []
# no_decay = ['bias', 'LayerNorm.weight']
# optimizer_grouped_parameters = [
#         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
#         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
# ]
# optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
# # scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma = 0.8)
# # total_steps = len(train_dataloader) * epochs

# # scheduler = get_linear_schedule_with_warmup(optimizer,
# #                                             num_warmup_steps=0,
# #                                             num_training_steps=total_steps)

# # Step_LR = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.8)

# 定义损失函数
criterion = nn.CrossEntropyLoss()


tb = SummaryWriter(log_dir = "runs/0317snli_lr1e5_maxlen256_wd0_th")
# tb.add_graph(model)

epochresult_list =[]
allstepresult_list = []

# 训练模型
for i in range(epochs):
    

    print(datetime.datetime.now(), "【======Epoch" + str(i+1) + "======】")
    
    # 训练模型
    epoch_num = i+1
    train_loss, train_correct, train_acc, stepresult_list= train(model, train_dataloader, test_dataloader, hardtest_dataloader, easytest_dataloader, device, num_labels, criterion,epoch_num)
    print("train loss: ", train_loss, "\t","train acc:", train_acc, "\t",  "train_correct:", train_correct )
    epochresult_list.append(i+1)
    epochresult_list.append('train result')
    epochresult_list.append(train_acc)
    
    allstepresult_list.append(stepresult_list)
    
    #把要在tensorboard中显示的数据加进来
    tb.add_scalar('train loss',train_loss, i+1)
    tb.add_scalar('train_correct',train_correct , i+1)
    tb.add_scalar('train_acc',train_acc, i+1)
    
    # 验证集看效果
    test_loss, test_correct, test_acc= test(model, test_dataloader, criterion, device)
    print("test loss: ", test_loss, "\t", "test acc:", test_acc, "\t", "test_correct:", test_correct)
    now_time = datetime.datetime.now()
    print(now_time, "【======test end======】")
    hardtest_loss, hardtest_correct, hardtest_acc= test(model, hardtest_dataloader, criterion, device)
    print("hardtest loss: ", hardtest_loss, "\t", "hardtest acc:", hardtest_acc, "\t", "hardtest_correct:", hardtest_correct)
    now_time = datetime.datetime.now()
    print(now_time, "【======test end======】")
    easytest_loss, easytest_correct, easytest_acc= test(model, easytest_dataloader, criterion, device)
    print("easytest loss: ", easytest_loss, "\t", "easytest acc:", easytest_acc, "\t", "easytest_correct:", easytest_correct)
    now_time = datetime.datetime.now()
    print(now_time, "【======test end======】")
    epochresult_list.append('test result')
    epochresult_list.append(test_acc)
    epochresult_list.append('hard test result')
    epochresult_list.append(hardtest_acc)
    epochresult_list.append('easy test result')
    epochresult_list.append(easytest_acc)
    
    
#         每个epoch保存一次模型
    print(datetime.datetime.now(), "【======保存Epoch" + str(i+1) + "的模型======】")
    model.save_pretrained('/data1/home/zmj/roberta/trained_model/duoceng12attention/trained_model/0317snli_lr1e5_maxlen128_wd0_th/epoch'+str(i+1))
    print(datetime.datetime.now(), "【======Epoch end======】")

print(epochresult_list)
print(allstepresult_list)

# 只保存最终模型
# model.save_pretrained('/data1/home/zmj/transformers/result/duoceng12/checkpoint-epoch12fb1w/out/roberta-duoceng'+str(i+1))