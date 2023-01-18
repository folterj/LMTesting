from transformers import pipeline

classifier = pipeline('text-classification', 'folterj/test-trainer')

result = classifier('This is incorrect sentence.')
print(result)
