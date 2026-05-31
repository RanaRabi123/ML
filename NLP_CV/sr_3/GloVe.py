import numpy as np
from scipy import spatial
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


embeddings_dict = {}
with open('glove.6B.50d.txt', 'r') as f:
    for line in f:
        values = line.split()
        word = values[0]
        vector = np.asarray(values[1:], dtype='float32')
        embeddings_dict[word] = vector


def closest_embedding(embedding, exclude=(), topn=1):
    return sorted((w for w in embeddings_dict if w not in exclude),
        key=lambda w: spatial.distance.euclidean(embeddings_dict[w], embedding))[:topn]


def analogy(a, b, c, topn=1):
    result_vec = embeddings_dict[b] - embeddings_dict[a] + embeddings_dict[c]
    return closest_embedding(result_vec, exclude={a, b, c}, topn=topn)


analogies = [
    ("man",     "king",    "woman"),    
    ("paris",   "france",  "berlin"),   
    ("walked",  "walking", "swam"),     
    ("good",    "better",  "bad"),      
    ("small",   "smaller", "large"),    
]

print("Word Analogies  (a : b  ::  c : ?)\n"+ "─"* 44)
for a, b, c in analogies:
    answer = analogy(a, b, c)[0]
    print(f"  {a:10} : {b:10} :: {c:10} --> {answer}")



tsne = TSNE(n_components=2, random_state=11)    # same as PCA , but good for visuals 

words   = list(embeddings_dict.keys())
vectors = [embeddings_dict[w] for w in words]

y = tsne.fit_transform(np.array(vectors[:100]))


plt.figure(figsize=(10, 10))
plt.scatter(y[:, 0], y[:, 1], s=10)

for i, word in enumerate(words[:100]):
    plt.annotate(word, (y[i, 0], y[i, 1]), fontsize=8)
plt.title("t-SNE of first 100 GloVe words")
plt.show()