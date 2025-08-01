---
title: Unsupervised text clustering using GNN (Graph Auto Encoder)
description: Unsupervised text clustering using GNN (Graph Auto Encoder).
date: 2025-06-18
tags: posts
---

<a target="_blank" href="https://colab.research.google.com/github/yonatanlou/notebooks/blob/main/unsupervised-gae-clustering.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

## Unsupervised text clustering using GNN

Graph neural networks (GNNs) have received a fair amount of attention over the past few years. That said, some of the initial excitement has faded-especially in certain research domains. Part of this decline is due to the rise of transformer models, which in many ways behave like fully connected GNNs. This has led some people to question whether GNNs are still relevant or necessary. ([Transformers vs GNNs – Taewoon Kim](https://taewoon.kim/2024-10-15-transformer-vs-gnn/), [Transformers are GNNs – Graph Deep Learning](https://graphdeeplearning.github.io/post/transformers-are-gnns/), [Reddit discussion: Are GNNs obsolete?](https://www.reddit.com/r/MachineLearning/comments/1jgwjjk/d_are_gnns_obsolete_because_of_transformers/)).


Personally, I still find GNNs extremely useful-particularly in two situations:

1. When your data naturally forms a graph.
2. When you want to combine multiple types of features in a "learnable" and flexible way.

In this post, I’ll walk through how to implement **unsupervised text clustering** using a **[Graph Autoencoder (GAE)](https://arxiv.org/abs/1611.07308)** framework that supports multiple feature types.

This is more of a quick-and-dirty prototype than a polished package. I wrote it mainly because I couldn’t find a simple example of unsupervised text clustering using GNNs online.

If you're looking for a more customizable and production-ready version, you can check out the [`QumranNLP`](https://github.com/yonatanlou/QumranNLP) repository. It's built around a fascinating dataset-texts from the Dead Sea Scrolls-and uses a more refined version of the same approach.


First of all, we will import some important libraries, make some constants (which can be optimize in the future), and collect the data.


```python
import torch, random, numpy as np
from tqdm.auto import tqdm
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from transformers import AutoTokenizer, AutoModel
from sklearn.cluster import KMeans
from sklearn.metrics import v_measure_score, adjusted_rand_score
from torch_geometric.nn import GAE, GCNConv
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected


SAMPLES = 1000           # subset size
N_CLUST = 20            # k‑means clusters (20‑news groups)
HIDDEN  = 256           # GCN hidden dim
LATENT  = 128           # GCN latent dim
LR      = 0.001         # learning rate
EPOCHS  = 350           # training epochs
DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
Q_SIM     = 0.999
SEED = 42


print("Step 1 | Loading 20‑Newsgroups …")
news = fetch_20newsgroups(remove=("headers", "footers", "quotes"))
texts, y = news.data[:SAMPLES], news.target[:SAMPLES]


def set_seed_globally(seed=42):
    # Set seeds
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed_globally(SEED)


```


Now, we'll represent the text using three different methods. These are just examples-you can easily swap them out or tweak the configurations to fit your own data or preferences.

The methods we'll use are:

- **BERT embeddings** – contextual representations from a pretrained language model.
- **TF-IDF** – a classic, sparse representation that captures term importance across the corpus.
- **Character n-grams** – helpful for capturing subword patterns, especially in noisy texts.



```python
print("Step 2 | DistilBERT embeddings …")
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model_bert = AutoModel.from_pretrained("distilbert-base-uncased").to(DEVICE).eval()

@torch.no_grad()
def bert_embed(docs, bs=16):
    out = []
    for i in tqdm(range(0, len(docs), bs), desc="Embedding", leave=False):
        batch = docs[i:i+bs]
        inp = tokenizer(batch, padding=True, truncation=True, max_length=512, return_tensors="pt").to(DEVICE)
        out.append(model_bert(**inp).last_hidden_state.mean(dim=1).cpu())
    return torch.cat(out).numpy()

Xb = bert_embed(texts)


print("Step 3 | TF‑IDF & char‑3‑gram …")
Xt = TfidfVectorizer(max_features=1500).fit_transform(texts).toarray()
Xn = CountVectorizer(analyzer="char", ngram_range=(3, 3), max_features=1500).fit_transform(texts).toarray()

```


At this point, we're building the graph based on the TF-IDF and character n-gram features.

There are quite a few parameters you can tweak here, and each choice can significantly affect your model. Some key considerations:

- **Similarity metric**: How do we calculate the similarity between vectors? Common options are cosine similarity and Euclidean distance.
- **Graph structure**: Do we want a **heterogeneous graph** (with multiple edge types, one for each feature type), or a **homogeneous graph** (a single adjacency matrix that combines all features)?

These decisions give you a lot of flexibility-and room for creativity-to improve your model.

> 🔧 **Note:** One of the most critical parameters is the similarity threshold for edge creation (`Q_SIM`).  
If this threshold is set too low, you’ll end up with a massive graph-which means you'll need a lot of GPU's just to train the model.  
Through trial and error, I’ve found that using a **higher threshold** often results in **better performance** and **faster training**.



```python
print("Step 4 | Building graph edges (k‑NN) …")

def adj_cosine(mat, q=0.99):
    norm = mat / (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    sim  = norm @ norm.T
    thresh = np.quantile(sim, q)
    adj = (sim >= thresh).astype(float)
    np.fill_diagonal(adj, 1)
    return adj

def adj_to_edge(adj):
    src, dst = np.nonzero(adj)
    return to_undirected(torch.tensor([src, dst], dtype=torch.long))

adj_tfidf = adj_cosine(Xt, Q_SIM)
adj_ngram = adj_cosine(Xn, Q_SIM)
adj_comb  = ((adj_tfidf + adj_ngram) > Q_SIM).astype(float)  # union
print(f"   TF-IDF edges:  {int(adj_tfidf.sum())}")
print(f"   N-gram edges: {int(adj_ngram.sum())}")
print(f"   Combined edges: {int(adj_comb.sum())}")
E = adj_to_edge(adj_comb)


```

```bash
Step 4 | Building graph edges (k‑NN)
    TF-IDF edges:  1032
    N-gram edges: 1028
    Combined edges: 1056
```

Now we move on to training the model.  
In my original implementation, I included early stopping to avoid overfitting-but for the sake of this simplified version, I skipped it (I was lazy 😅).

Just like before, this part is highly customizable. You can experiment with:

- The number of layers
- Hidden dimensions
- Dropout rates
- Batch normalization
- Activation functions

Feel free to design the GAE/VGAE architecture in a way that fits your data and goals.

---

Evaluating unsupervised clustering models is always a bit of a mystery. There's no single "correct" metric, and depending on your application, some may be more meaningful than others.  
Still, I think [Scikit-learn’s guide on Clustering Performance Evaluation](https://scikit-learn.org/stable/modules/clustering.html#clustering-performance-evaluation) is one of the best overviews available online.

I also wrote about a more niche but useful method in my post on  
[Evaluating Hierarchical Clustering](https://yonatanlou.github.io/blog/Evaluating-Hierarchical-Clustering/hierarchical-clustering-eval/), which dives into metrics like the Dasgupta cost (specific for hierarchial clustering).



```python

print("Step 5 | Training Graph Auto‑Encoder …")

graph = Data(x=torch.tensor(Xb, dtype=torch.float), edge_index=E)
graph = graph.to(DEVICE)

class Encoder(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_z):
        super().__init__()
        self.g1 = GCNConv(dim_in, dim_h)
        self.g2 = GCNConv(dim_h, dim_z)
    def forward(self, x, ei):
        return self.g2(self.g1(x, ei).relu(), ei)

gae = GAE(Encoder(graph.x.size(1), HIDDEN, LATENT)).to(DEVICE)
opt = torch.optim.Adam(gae.parameters(), lr=LR)

for epoch in range(1, EPOCHS + 1):
    gae.train(); opt.zero_grad()
    z = gae.encode(graph.x, graph.edge_index)
    loss = gae.recon_loss(z, graph.edge_index)
    loss.backward(); opt.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f}")



```

```bash
    Step 5 | Training Graph Auto‑Encoder …
    Epoch 010 | Loss 1.1351
    Epoch 020 | Loss 1.0478
    Epoch 030 | Loss 0.9715
    Epoch 040 | Loss 0.8702
    Epoch 050 | Loss 0.8832
    Epoch 060 | Loss 0.8712
    Epoch 070 | Loss 0.8518
    Epoch 080 | Loss 0.8291
    Epoch 090 | Loss 0.8149
    Epoch 100 | Loss 0.7946
    Epoch 110 | Loss 0.8166
    Epoch 120 | Loss 0.8010
    Epoch 130 | Loss 0.7978
    Epoch 140 | Loss 0.7979
    Epoch 150 | Loss 0.8014
    Epoch 160 | Loss 0.8089
    Epoch 170 | Loss 0.7826
    Epoch 180 | Loss 0.7878
    Epoch 190 | Loss 0.8120
    Epoch 200 | Loss 0.7809
    Epoch 210 | Loss 0.7806
    Epoch 220 | Loss 0.7765
    Epoch 230 | Loss 0.7945
    Epoch 240 | Loss 0.7801
    Epoch 250 | Loss 0.7783
    Epoch 260 | Loss 0.7951
    Epoch 270 | Loss 0.7917
    Epoch 280 | Loss 0.7733
    Epoch 290 | Loss 0.7740
    Epoch 300 | Loss 0.7602
    Epoch 310 | Loss 0.7769
    Epoch 320 | Loss 0.7720
    Epoch 330 | Loss 0.7843
    Epoch 340 | Loss 0.7836
    Epoch 350 | Loss 0.7654

```


```python
print("Step 6 | Clustering latent space …")
gae.eval()
with torch.no_grad():
    embeddings = gae.encode(graph.x, graph.edge_index).cpu().detach().numpy()
    km_emb  = KMeans(N_CLUST, n_init=10).fit(embeddings)
    gae_v   = v_measure_score(y, km_emb.labels_)
    gae_ari = adjusted_rand_score(y, km_emb.labels_)

km_base = KMeans(N_CLUST, n_init=10).fit(Xb)
base_v   = v_measure_score(y, km_base.labels_)
base_ari = adjusted_rand_score(y, km_base.labels_)



print("\nResults")
print(f"Baseline (BERT + k‑means) → V: {base_v:.3f} | ARI: {base_ari:.3f}")
print(f"GAE (BERT + TF-IDF + N-grams)   → V: {gae_v:.3f} | ARI: {gae_ari:.3f}")
print(f"Improvement: V {(gae_v/base_v)-1:.3%} | ARI {(gae_ari/base_ari)-1:.3%}")
```
```bash
    Step 6 | Clustering latent space …
    
    Results
    Baseline (BERT + k‑means) → V: 0.317 | ARI: 0.124
    GAE (BERT + TF-IDF + N-grams)   → V: 0.355 | ARI: 0.153
    Improvement: V 12.082% | ARI 23.501%
```

```bibtex
@misc{lou2025gae-Unsupervised-text-clustering-using-GNN,
  author       = {Yonatan Lourie},
  title        = {Evaluating Hierarchical Clustering Beyond the Leaves},
  year         = {2025},
  howpublished = {\url{https://yonatanlou.github.io/blog/unsupervised-text-clustering-gnn/unsupervised-gae-clustering/}},
  note         = {Accessed: 2025-06-20}
}
```
