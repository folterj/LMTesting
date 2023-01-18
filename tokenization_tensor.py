import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize(tokenizer, text):
    tokens = tokenizer.tokenize(text)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return ids


text = ["I’ve been waiting for a HuggingFace course my whole life.",
        "I hate this so much!"]

ids1 = tokenize(tokenizer, text[0])
ids2 = tokenize(tokenizer, text[1])

maxlen = max([len(ids1), len(ids2)])

ids = [ids1 + [tokenizer.pad_token_id] * (maxlen - len(ids1)), ids2 + [tokenizer.pad_token_id] * (maxlen - len(ids2))]
attention = [[1] * len(ids1) + [0] * (maxlen - len(ids1)), [1] * len(ids2) + [0] * (maxlen - len(ids2))]

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
print(ids1)
print(model(torch.tensor([ids1])).logits)
print(ids2)
print(model(torch.tensor([ids2])).logits)
print(model(torch.tensor(ids), attention_mask=torch.tensor(attention)).logits)

tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
print(tokens['input_ids'])
print(model(**tokens))
