from datasets import load_dataset

base_url = "https://rajpurkar.github.io/SQuAD-explorer/dataset/"
data_files = {
    "train": base_url + "train-v2.0.json",
    "test": base_url + "dev-v2.0.json",
}

squad_dataset = load_dataset("json", data_files=data_files, field="data")
