import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# word = i love ai
d_k = 32  # embedding dimension

np.random.seed(3)
word_1 = np.random.rand(1, d_k)
word_2 = np.random.rand(1, d_k)
word_3 = np.random.rand(1, d_k)

matrix_q = np.random.random(size=(d_k, d_k))
matrix_k = np.random.random(size=(d_k, d_k))
matrix_v = np.random.random(size=(d_k, d_k))

words = [word_1, word_2, word_3]
Q = [w @ matrix_q for w in words]
K = [w @ matrix_k for w in words]
V = [w @ matrix_v for w in words]

# ----------------------


def softmax(x):
    e_x = np.exp(x - np.max(x)) 
    return e_x / e_x.sum

n = len(words)
score_matrix = np.array([
    [( Q[i] @ K[j].T ).item() / np.sqrt(d_k) for j in range(n)]
    for i in range(n)
])  # shape: (3, 3)


weight_matrix = np.array([softmax(score_matrix[i]) for i in range(n)])


        #  final embeddings
attention_out = np.vstack([
    sum(weight_matrix[i][j] * V[j] for j in range(n))
    for i in range(n)
])  # shape: (3, d_k)




            #  visualizations of weights 
            #  weights , after passing from softmax
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

sns.heatmap(weight_matrix, annot=True, fmt='.2f', ax=axes[0])
axes[0].set_title('Attention weights (row = query, col = key)')
axes[0].set_xlabel('Key (attended to)')
axes[0].set_ylabel('Query (attending from)')

            #  before passing to softmax
sns.heatmap(score_matrix, annot=True, fmt='.2f', ax=axes[1])
axes[1].set_title('Raw scaled scores (before softmax)')
axes[1].set_xlabel('Key')
axes[1].set_ylabel('Query')
plt.show()