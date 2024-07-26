

def load_hf_token(filename='hf_token'):
    return open(filename, mode='r').read()


def remove_duplicate_sentences(text):
    output_text = ''
    last_line = None
    for line in text.splitlines():
        if line != last_line:
            if len(output_text) > 0:
                output_text += '\n'
            output_text += line
        last_line = line
    return output_text
