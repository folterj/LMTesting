import numpy as np
from datasets import load_dataset
import evaluate
from transformers import AutoTokenizer, DataCollatorWithPadding, TrainingArguments, AutoModelForSequenceClassification, \
    Trainer


raw_datasets = load_dataset("glue", "mrpc")
checkpoint = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def tokenize_function(example):
    return tokenizer(example["sentence1"], example["sentence2"], truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    metrics = metric.compute(predictions=predictions, references=labels)
    accuracies.append(metrics['accuracy'])
    f1scores.append(metrics['f1'])
    return metrics


tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

metrics_interval = 100

training_args = TrainingArguments("test-trainer",
                                  evaluation_strategy='steps', eval_steps=metrics_interval,
                                  log_level='info', logging_strategy='steps', logging_steps=metrics_interval)

model = AutoModelForSequenceClassification.from_pretrained(checkpoint, num_labels=2)
metric = evaluate.load("glue", "mrpc")
accuracies = []
f1scores = []

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    data_collator=data_collator,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
print(predictions.metrics)

print('Accuracy:', accuracies)
print('F1:', f1scores)

trainer.save_model('test_model')
