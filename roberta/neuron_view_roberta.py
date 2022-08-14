from bertviz.transformers_neuron_view import RobertaModel, RobertaTokenizer
from bertviz.neuron_view import show
from transformers import RobertaTokenizer, RobertaForSequenceClassification, RobertaConfig, AdamW
learning_rate = 1e-5
weight_decay = 1e-2
hidden_dropout_prob = 0.1
epochs = 6
batch_size = 16
max_len = 256 #并不是越长越好，样本数据集的长度最长是78
model_name ="/data1/home/zmj/roberta/trained_model/snli_lr1e5_maxlen128_wd1e2_3/epoch4step20000/"
num_labels = 3
output_hidden_states=True
output_attentions=True

model_type = 'roberta'
model_version = 'roberta-base'
# config = RobertaConfig.from_pretrained("roberta-base", num_labels=num_labels,output_hidden_states=output_hidden_states, output_attentions=output_attentions, hidden_dropout_prob=hidden_dropout_prob)
# model_version ="/data1/home/zmj/roberta/trained_model/snli_lr1e5_maxlen256_wd0/epoch3/"
model = RobertaModel.from_pretrained(model_version)
tokenizer = RobertaTokenizer.from_pretrained(model_version)
sentence_a = "Island native fishermen reeling in their nets after a long day's work."
sentence_b = "The men caught many fish."
# show(model, model_type, tokenizer, sentence_a, sentence_b)
show(model, model_type, tokenizer, sentence_a, sentence_b)