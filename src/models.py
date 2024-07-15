import torch
from transformers import AutoTokenizer, BertConfig, BertModel


# Building the model from the config
#model = BertModel(BertConfig())

# pre-trained
model_name = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertModel.from_pretrained(model_name)

sequences = ["Hello!", "Cool.", "Nice!"]
inputs = tokenizer(sequences, padding=True, truncation=True, return_tensors="pt")
outputs = model(**inputs)
