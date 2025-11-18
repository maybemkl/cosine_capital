import numpy as np

from collections import Counter

from nltk.corpus import gutenberg
from nltk.tokenize import RegexpTokenizer

def tokens_from_gutenberg_text(txt_name):
    sents = gutenberg.sents(txt_name)
    words = [word for sent in sents for word in sent]
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(" ".join(words).lower())
    return tokens
  
def get_probs_from_counter(counts):
    sum_ = sum(counts.values())
    probs = Counter()
    for token in counts:
        probs[token] = np.float32(np.format_float_positional(counts[token] / sum_))
    return probs