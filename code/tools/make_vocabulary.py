import argparse
from tqdm import tqdm
import numpy as np

parser = argparse.ArgumentParser(description='Train (and test) a model with a given block.')
parser.add_argument('--sentences')
parser.add_argument('--embeddings')
args = parser.parse_args()

unk_token = "<unk>"
extra_tokens = ["<s>", "</s>", unk_token]
embedding_width = 300

word_counts = {}

def tokenize(sentence):
    return sentence.split(" ")

with open(args.sentences, 'r') as sentences:
    for sentence in tqdm(list(sentences), desc="Counting words:"):
        for token in tokenize(sentence.strip()):
            if token not in word_counts:
                word_counts[token] = 1
            else:
                word_counts[token] += 1

def get_top_n_tokens(count_dictionary, n):
    sorted_by_value = sorted(count_dictionary.items(), key=lambda kv: kv[1], reverse=True)
    kept = sorted_by_value[:n]
    leftover = sorted_by_value[n:]
    to_return = [kv[0] for kv in kept]
    leftover_to_return = [kv[0] for kv in leftover]
    return to_return, leftover_to_return

def get_tokens_mentioned_n_times(count_dictionary, n):
    to_keep = [kv[0] for kv in count_dictionary.items() if kv[1] >= n]
    leftover = [kv[0] for kv in count_dictionary.items() if kv[1] < n]

    return to_keep, leftover

kept_tokens, leftover_tokens = get_tokens_mentioned_n_times(word_counts, 3)
all_tokens = extra_tokens + kept_tokens

word_to_index_dict = {w:idx for idx, w in enumerate(all_tokens)}

vocabulary_embeddings = np.random.uniform(-0.01, 0.01, (len(all_tokens), embedding_width)).astype(np.float32)

with open(args.embeddings, 'r') as embeddings:
    for word_embedding in tqdm(list(embeddings), desc="Reading embeddings:"):
        parts = word_embedding.split(" ")

        word = parts[0]

        if word in word_to_index_dict:
            idx = word_to_index_dict[word]
            vocabulary_embeddings[idx] = np.array(parts[1:], dtype=np.float32)

vocabulary_embeddings = vocabulary_embeddings.astype(np.str)

for i, token in enumerate(all_tokens):
    if i > 0:
        print("")
    print(token, end=" ")
    print(" ".join(vocabulary_embeddings[i]), end="")

for token in leftover_tokens:
    print("")
    print(token, end=" ")
    print(" ".join(vocabulary_embeddings[word_to_index_dict[unk_token]]), end="")
