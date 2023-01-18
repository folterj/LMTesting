import requests
from datasets import load_dataset, Features, Value


def get_body(url):
    response = requests.get(url, headers=headers)
    return [r["body"] for r in response.json()]


def get_comments(issue_number):
    url = f"https://api.github.com/repos/huggingface/datasets/issues/{issue_number}/comments"
    return get_body(url)


with open('github_token.txt') as file:
    GITHUB_TOKEN = file.read()
    headers = {"Authorization": f"token {GITHUB_TOKEN}"}




#features = Features({'closed_at': Value(dtype='timestamp[s]', id=None)})
#issues_dataset = load_dataset("json", data_files="issues/datasets-issues.jsonl", split="train", features=features)
issues_dataset = load_dataset("json", data_files="issues/datasets-issues.jsonl", split="train", streaming=True)

samples = issues_dataset.shuffle(42).take(3)

# Print out the URL and pull request entries
for sample in samples:
    print(f">> URL: {sample['url']}")
    print(f">> Pull request: {sample['pull_request']}")
    print(f">> Comments: {get_body(sample['comments_url'])}")
    print()

issues_dataset = issues_dataset.map(
    lambda x: {"is_pull_request": False if x["pull_request"] is None else True}
)

issues_dataset = list(issues_dataset)

print(get_comments(2792))
