from transformers import AutoTokenizer


def tokenize(tokenizer_name, text):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokens = tokenizer.tokenize(text)
    return tokens


text = "Using a Transformer network is simple"
print(tokenize("bert-base-cased", text))
print(tokenize("bert-base-uncased", text))
print(tokenize("albert-base-v1", text))


tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
tokens = tokenizer.tokenize(text)
ids = tokenizer.convert_tokens_to_ids(tokens)
print(ids)
text2 = tokenizer.decode(ids)
print(text2)
