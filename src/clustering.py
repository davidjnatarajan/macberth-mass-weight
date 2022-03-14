from collections import defaultdict
from iteration_utilities import deepflatten
from matplotlib.pylab import figure, imshow, show
from pandas import read_csv
from seaborn import heatmap, scatterplot, color_palette
from sklearn.metrics import pairwise_distances
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer
import sys
import torch


def get_token_indexes(ids, tokenizer, prefix='##'):
    output = defaultdict(list)
    special = set(tokenizer.special_tokens_map.values())
    subwords = tokenizer.convert_ids_to_tokens(ids)
    ids, word = [], ''
    for idx, subword in enumerate(subwords):
        if subword in special:
            continue
        if subword.startswith(prefix):
            word += subword[len(prefix):]
            ids.append(idx)
        else:
            if word:
                output[word].append(ids)
            ids, word = [idx], subword
    if word:
        output[word].append(ids)
    return output

m = AutoModel.from_pretrained('emanjavacas/MacBERTh', torch_dtype=torch.float16)
tok = AutoTokenizer.from_pretrained('emanjavacas/MacBERTh')

data = read_csv('./data/thesis-utf8-data.csv')

a, b, c = zip(*data[['Context before', 'Query term', 'Context after']].values)

sents, keywords = [], []
for a1, b1, c1 in zip(a, b, c):
    sents.append(a1 + ' ' + b1 + ' '+ c1)
    keywords.append(b1)

ids = tok(list(sents), return_tensors='pt', padding=True)
output = m(**ids)

# target = []
# for idx, keyword in enumerate(keywords):
#     mapping = get_token_indexes(ids.input_ids[idx], tok)
#     current = mapping[keyword]
#     if len(current) > 1:
#         target.append(current[0])
#     else :
#         target.append(mapping[keyword])
#     #print(target, "--", keyword, "--", sents[idx])
# target = list(deepflatten(target))

# np_arr = output['last_hidden_state'].cpu().detach().numpy()
# embeddings = []
# for i in list(range(1500)):
#     embeddings.append(np_arr[i][target[i]].tolist())

# pca_embeddings = PCA(n_components=50).fit_transform(embeddings)
# tsne_embeddings = TSNE(perplexity=250, init='pca', random_state=153).fit_transform(pca_embeddings)

# distance = pairwise_distances(tsne_embeddings)
# ax = heatmap(distance, linewidth=0)
# show()

# categories = list(data['Sense'])

# figure(figsize=(16,10))
# scatterplot(
#     x=tsne_embeddings[:,0], y=tsne_embeddings[:,1],
#     hue=categories,
#     palette=color_palette('Set2', len(set(categories))),
#     legend="full",
# )