from datasets import load_dataset
from transformers import AutoTokenizer


# Download & extraction the baseline data w/o streaming takes various minutes
#url = "https://the-eye.eu/public/AI/pile_preliminary_components/PUBMED_title_abstracts_2019_baseline.jsonl.zst"
url = "https://the-eye.eu/public/AI/pile/"

if url.endswith('/'):
    data_files = {
        "train": [url + "train/" + f"{idx:02d}.jsonl.zst" for idx in range(30)],
        "validation": url + "val.jsonl.zst",
        "test": url + "test.jsonl.zst",
    }
else:
    data_files = url

# Use streaming!
dataset = load_dataset("json", data_files=data_files, streaming=True)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
tokenized_dataset = dataset.map(lambda x: tokenizer(x["text"]))
print(next(iter(tokenized_dataset['train'])))
