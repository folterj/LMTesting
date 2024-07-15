# UCI Machine Learning Repository: https://archive.ics.uci.edu/

import numpy as np
import pandas as pd
from datasets import load_dataset
#from transformers import AutoModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"

print('Loading dataset')
dataset0 = load_dataset("csv", data_files=url, sep=";")['train']
dataset = dataset0.train_test_split(test_size=0.2)
train_labels = dataset['train']['quality']
train_features = dataset['train'].remove_columns('quality')
test_labels = dataset['test']['quality']
test_features = dataset['test'].remove_columns('quality')

print('Creating model')
#model = AutoModel.from_config('tabular-classification')
model = RandomForestClassifier(n_jobs=4)
print('Training model')
model.fit(train_features.to_pandas(), train_labels)

print('Evaluating model')
predictions = model.predict(test_features.to_pandas())
f1_score = f1_score(test_labels, predictions, average='macro', zero_division=0)

print(f'F1 score: {f1_score:.3f}')
print('Label/predictions:', *zip(test_labels, predictions))
print(f'Mean error in prediction of quality value: '
      f'{np.mean([abs(np.diff(pair)) for pair in zip(test_labels, predictions)]):.3f}')
