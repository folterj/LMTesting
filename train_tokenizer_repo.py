import evaluate
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, TrainingArguments, AutoModelForSequenceClassification, Trainer
from huggingface_hub import login


login()


raw_datasets = load_dataset("glue", "cola")
checkpoint = "bert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)


def preprocess_function(examples):
    return tokenizer(examples["sentence"], truncation=True)


def compute_metrics(eval_preds):
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)

training_args = TrainingArguments("test-trainer",
                                  num_train_epochs=3,
                                  learning_rate=2e-5,
                                  weight_decay=0.01,
                                  save_strategy='epoch',
                                  evaluation_strategy='epoch',
                                  push_to_hub=True,
                                  )

model = AutoModelForSequenceClassification.from_pretrained(checkpoint)

metric = evaluate.load("glue", "cola")

trainer = Trainer(
    model,
    training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

predictions = trainer.predict(tokenized_datasets["validation"])
print(predictions.predictions.shape, predictions.label_ids.shape)
print(predictions.metrics)

trainer.push_to_hub('End of training')

label_names = raw_datasets['train'].features['label'].names
model.config.id2label = {str(i): label for i, label in enumerate(label_names)}
model.config.label2id = {label: str(i) for i, label in enumerate(label_names)}
model.config.push_to_hub(training_args.output_dir)
