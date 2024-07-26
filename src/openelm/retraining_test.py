# source: https://huggingface.co/apple/OpenELM-270M

import numpy as np
import torch
import evaluate
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, pipeline

from src.utils import *


#output = generate(
#   model='apple/OpenELM-270M',
#   hf_access_token=hf_access_token,
#   prompt=prompt,
#   generate_kwargs=generate_kwargs)


class ModelTest:
    def __init__(self, model_name, tokenizer_name, max_length=1024, params={}):
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name
        self.max_length = max_length
        self.params = params
        self.hf_access_token = load_hf_token()
        if torch.cuda.is_available() and torch.cuda.device_count():
            self.device = 'cuda:0'
            print(f'Running on GPU ({torch.cuda.get_device_name()})')
        else:
            self.device = 'cpu'
            print('Running on CPU')

    def train(self):
        print('Start training')
        self.model.train()

        self.tokenizer.pad_token = self.tokenizer.eos_token

        def preprocess_function(data):
            # TODO: all lists in batch must have same # of elements; fix padding - padding not working!
            model_inputs = self.tokenizer(text=data['article'], max_length=1024, truncation=True, padding=True)
            labels = self.tokenizer(text_target=data['highlights'], max_length=1024, truncation=True, padding=True)
            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        def compute_metrics(eval_preds):
            logits, labels = eval_preds
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir="test-trainer",
            eval_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=4,
            fp16=True
        )

        print('Loading training data')
        raw_datasets = load_dataset("abisee/cnn_dailymail", "3.0.0")

        print('Preprocessing training data')
        tokenized_datasets = raw_datasets.map(preprocess_function, batched=True)
        #tokenized_datasets = [preprocess_function(data) for data in tqdm(raw_datasets['train'])]

        print('Training')
        metric = evaluate.load("rouge")
        trainer = Trainer(
            self.model,
            training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()

    def create(self):
        print('Creating model')
        self.model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)
        self.model.to(self.device)
        print('Creating tokenizer')
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, token=self.hf_access_token)

    def evaluate(self, prompt):
        prompt_encoded = self.tokenizer(prompt)
        prompt_tensor = torch.tensor(prompt_encoded['input_ids'], device=self.device).unsqueeze(0)

        self.model.eval()
        output_ids = self.model.generate(prompt_tensor, max_length=self.max_length, pad_token_id=0, **self.params)
        output_text = remove_duplicate_sentences(self.tokenizer.decode(output_ids[0], skip_special_tokens=True))
        return output_text

    def summarize(self, text):
        task_pipeline = pipeline(model=self.model, tokenizer=self.tokenizer, task="summarization", device=self.device)
        output = task_pipeline(text, max_new_tokens=1024)
        return output

    def retokenize_sentences(self, text):
        new_line = '<0x0A>'
        tokens = self.tokenizer.tokenize(text)
        indices = [index for index, token in enumerate(tokens) if token == '.']
        tokens_sents = np.split(tokens, np.array(indices) + 1)
        new_tokens = []
        for sent_index, tokens_sent in enumerate(tokens_sents):
            tokens_sent = tokens_sent.tolist()
            tokens_sent[0] = tokens_sent[0].lstrip('▁')
            if sent_index > 0 and tokens_sent[0] != new_line:
                tokens_sent.insert(0, new_line)
        output_text = self.tokenizer.convert_tokens_to_string(new_tokens)
        return output_text


if __name__ == '__main__':
    model_name = 'apple/OpenELM-270M'
    tokenizer_name = 'meta-llama/Llama-2-7b-hf'
    max_length = 1024
    params = {
        'repetition_penalty': 1.2
    }

    model = ModelTest(model_name, tokenizer_name, max_length, params=params)
    model.create()

    #print(model.evaluate('Once upon a time there was'))

    example = "Many diffusion systems share the same components, allowing you to adapt a pretrained model for one task to an entirely different task. This guide will show you how to adapt a pretrained text-to-image model for inpainting by initializing and modifying the architecture of a pretrained."
    #print(model.summarize(example))

    model.train()
    print(model.summarize(example))
