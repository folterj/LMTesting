from torch.utils.data import DataLoader
from transformers import AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset


checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)
raw_datasets = load_dataset("glue", "mrpc")
print(raw_datasets)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets = tokenized_datasets.remove_columns(raw_datasets["train"].column_names)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


samples = tokenized_datasets["train"][:8]
samples = {k: v for k, v in samples.items() if k not in ["idx", "sentence1", "sentence2"]}
print([len(x) for x in samples["input_ids"]])

batch = data_collator(samples)
print({k: v.shape for k, v in batch.items()})


train_dataloader = DataLoader(
    tokenized_datasets['train'], batch_size=16, shuffle=True, collate_fn=data_collator
)

for step, batch in enumerate(train_dataloader):
    print(batch['input_ids'].shape)
    if step > 5:
        break
