from bertviz.transformers_neuron_view import BertModel, BertTokenizer
from bertviz.neuron_view import show

model_type = 'bert'
model_version = 'bert-base-uncased'
do_lower_case = True
model = BertModel.from_pretrained(model_version)
tokenizer = BertTokenizer.from_pretrained(model_version, do_lower_case=do_lower_case)
sentence_a = "A land rover is being driven across a river."
sentence_b = "A sedan is stuck in the middle of a river."

show(model, model_type, tokenizer, sentence_a, sentence_b, display_mode='dark', layer=2, head=0)