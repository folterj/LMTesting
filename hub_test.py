from huggingface_hub import login, upload_file
login()

with open('test.md', 'w+') as f:
    f.write('test model')

upload_file(path_or_fileobj='test.md', path_in_repo='test.md', repo_id='folterj/test')
