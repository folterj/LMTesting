from huggingface_hub import Repository
from transformers import AutoTokenizer, AutoModelForSequenceClassification

repo = Repository('test', clone_from='folterj/test')
repo.git_pull()

tokenizer = AutoTokenizer.from_pretrained('test_model')
model = AutoModelForSequenceClassification.from_pretrained('test_model')
tokenizer.save_pretrained(repo.local_dir)
model.save_pretrained(repo.local_dir)

repo.git_add()
repo.git_commit('Added test model')
repo.git_push()
