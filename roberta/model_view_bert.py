from bertviz import model_view
from transformers import AutoTokenizer, AutoModel, utils

utils.logging.set_verbosity_error()  # Suppress standard warnings

# NOTE: This code is model-specific

model_version = 'bert-base-uncased'
sentence_a = "A land rover is being driven across a river."
sentence_b = "A vehicle is crossing a river."

model = AutoModel.from_pretrained(model_version, output_attentions=True)
tokenizer = AutoTokenizer.from_pretrained(model_version)
inputs = tokenizer.encode_plus(sentence_a, sentence_b, return_tensors='pt')
input_ids = inputs['input_ids']
token_type_ids = inputs['token_type_ids'] # token type id is 0 for Sentence A and 1 for Sentence B
attention = model(input_ids, token_type_ids=token_type_ids)[-1]
sentence_b_start = token_type_ids[0].tolist().index(1) # Sentence B starts at first index of token type id 1
token_ids = input_ids[0].tolist() # Batch index 0
tokens = tokenizer.convert_ids_to_tokens(token_ids)    
model_view(attention, tokens, sentence_b_start)