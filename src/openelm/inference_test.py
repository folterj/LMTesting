# source: https://huggingface.co/apple/OpenELM-270M

from src.openelm.generate_openelm import generate
from src.utils import *


prompt = 'Once upon a time there was'
generate_kwargs = {
    'repetition_penalty': 1.2
}
hf_access_token = load_hf_token()

output = generate(model='apple/OpenELM-270M', hf_access_token=hf_access_token, prompt=prompt, generate_kwargs=generate_kwargs)
