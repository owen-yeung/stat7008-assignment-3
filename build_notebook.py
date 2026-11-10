#!/usr/bin/env python3
"""Build STAT7008A Assignment 3 Jupyter notebook."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

code("""import warnings
warnings.filterwarnings('ignore')

import os
import re
import time
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from bs4 import BeautifulSoup
from PIL import Image
from IPython.display import display

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import MiniBatchKMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, silhouette_score, ConfusionMatrixDisplay
)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
torch.manual_seed(RANDOM_STATE)

os.makedirs('figures', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

sns.set_theme(style='whitegrid', context='notebook')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 150
plt.rcParams['figure.figsize'] = (8, 5)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'PyTorch device: {DEVICE}')
print(f'NumPy {np.__version__}, Pandas {pd.__version__}, Torch {torch.__version__}')
""")

# ============================================================
# Q1
# ============================================================
md("""# 1.
## 1(a)
""")

code("""# Fashion MNIST class names (OpenML / torchvision order)
FASHION_LABELS = {
    0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat',
    5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8: 'Bag', 9: 'Ankle boot'
}

print('Loading Fashion MNIST via torchvision...')
_train = torchvision.datasets.FashionMNIST(root='data', train=True, download=True)
_test = torchvision.datasets.FashionMNIST(root='data', train=False, download=True)

X_train_full = _train.data.numpy().reshape(len(_train), -1).astype(np.float32)
y_train_full = np.array(_train.targets, dtype=int)
X_test = _test.data.numpy().reshape(len(_test), -1).astype(np.float32)
y_test = np.array(_test.targets, dtype=int)
X_all = np.vstack([X_train_full, X_test])
y_all = np.concatenate([y_train_full, y_test])

print('=' * 60)
print('DATA VOLUME')
print('=' * 60)
print(f'Total samples      : {len(X_all):,}')
print(f'Training samples   : {len(X_train_full):,}')
print(f'Test samples       : {len(X_test):,}')
print(f'Feature dimension  : {X_train_full.shape[1]} (28x28 flattened)')
print(f'Number of classes  : {len(np.unique(y_all))}')
print(f'Pixel value range  : [{X_all.min():.0f}, {X_all.max():.0f}]')
print(f'Data type          : {X_all.dtype}')
""")

code("""# Display basic data samples — one example per class
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle('Fashion MNIST — Sample Images by Class', fontsize=14, fontweight='bold')

for cls in range(10):
    idx = np.where(y_train_full == cls)[0][0]
    ax = axes[cls // 5, cls % 5]
    ax.imshow(X_train_full[idx].reshape(28, 28), cmap='gray')
    ax.set_title(f'{cls}: {FASHION_LABELS[cls]}', fontsize=10)
    ax.axis('off')

plt.tight_layout()
plt.savefig('figures/q1a_samples.png', bbox_inches='tight')
plt.show()
""")

md("""Fashion MNIST is balanced — 6,000 images per class in training and 1,000 per class in test.
""")

code("""# Sample distribution of different classes
train_counts = pd.Series(y_train_full).value_counts().sort_index()
test_counts = pd.Series(y_test).value_counts().sort_index()

dist_df = pd.DataFrame({
    'Class': [FASHION_LABELS[i] for i in range(10)],
    'Train': train_counts.values,
    'Test': test_counts.values
})
print('Class Distribution:')
display(dist_df)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(range(10), train_counts.values, color='steelblue', edgecolor='black')
axes[0].set_xticks(range(10))
axes[0].set_xticklabels([FASHION_LABELS[i] for i in range(10)], rotation=45, ha='right')
axes[0].set_title('Training Set Class Distribution')
axes[0].set_ylabel('Count')
axes[0].set_xlabel('Class')

axes[1].bar(range(10), test_counts.values, color='coral', edgecolor='black')
axes[1].set_xticks(range(10))
axes[1].set_xticklabels([FASHION_LABELS[i] for i in range(10)], rotation=45, ha='right')
axes[1].set_title('Test Set Class Distribution')
axes[1].set_ylabel('Count')
axes[1].set_xlabel('Class')

plt.tight_layout()
plt.savefig('figures/q1a_class_distribution.png', bbox_inches='tight')
plt.show()
""")

md("""## 1(b)
""")

code("""# Normalize for clustering (0-1 scale)
scaler_km = MinMaxScaler()
X_train_scaled = scaler_km.fit_transform(X_train_full)
X_test_scaled = scaler_km.transform(X_test)

# MiniBatchKMeans with K=10 (true number of categories, for initial exploration)
K_TRUE = 10
print(f'Running MiniBatchKMeans with K={K_TRUE} on {len(X_train_scaled):,} training samples...')
t0 = time.time()
mbk_init = MiniBatchKMeans(
    n_clusters=K_TRUE,
    random_state=RANDOM_STATE,
    batch_size=2048,
    n_init=10,
    max_iter=100
)
cluster_labels_init = mbk_init.fit_predict(X_train_scaled)
print(f'Done in {time.time() - t0:.1f}s | Inertia: {mbk_init.inertia_:,.0f}')

# t-SNE on a subsample for tractable visualization
TSNE_N = 5000
tsne_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train_scaled), TSNE_N, replace=False)
X_tsne_sample = X_train_scaled[tsne_idx]
y_tsne_true = y_train_full[tsne_idx]
y_tsne_cluster = cluster_labels_init[tsne_idx]

print(f'Running t-SNE on {TSNE_N} samples...')
t0 = time.time()
tsne = TSNE(n_components=2, random_state=RANDOM_STATE, perplexity=30, max_iter=1000, init='pca')
X_embedded = tsne.fit_transform(X_tsne_sample)
print(f't-SNE done in {time.time() - t0:.1f}s')
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Colored by true labels
sc0 = axes[0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne_true,
                      cmap='tab10', s=8, alpha=0.7)
axes[0].set_title('t-SNE — Colored by True Labels')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')
cbar0 = plt.colorbar(sc0, ax=axes[0], ticks=range(10))
cbar0.ax.set_yticklabels([FASHION_LABELS[i] for i in range(10)])

# Colored by cluster assignment
sc1 = axes[1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne_cluster,
                      cmap='tab10', s=8, alpha=0.7)
axes[1].set_title(f't-SNE — Colored by MiniBatchKMeans (K={K_TRUE})')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')
plt.colorbar(sc1, ax=axes[1], ticks=range(K_TRUE), label='Cluster')

plt.tight_layout()
plt.savefig('figures/q1b_tsne_clusters.png', bbox_inches='tight')
plt.show()
""")

md("""## 1(c)
""")

code("""# --- Elbow Method ---
K_RANGE = range(2, 16)
inertias = []
silhouette_scores = []

print('Evaluating MiniBatchKMeans for K = 2..15 ...')
for k in K_RANGE:
    mbk = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                          batch_size=2048, n_init=5, max_iter=100)
    labels_k = mbk.fit_predict(X_train_scaled)
    inertias.append(mbk.inertia_)
    # Silhouette on a subsample (full set is expensive)
    sil_idx = np.random.RandomState(k).choice(len(X_train_scaled), 8000, replace=False)
    sil = silhouette_score(X_train_scaled[sil_idx], labels_k[sil_idx], sample_size=None)
    silhouette_scores.append(sil)
    print(f'  K={k:2d} | Inertia={mbk.inertia_:,.0f} | Silhouette={sil:.4f}')
""")

md("""Inertia decreases with K; a bend often appears near K=4–6, while the silhouette score typically peaks at a smaller K than the true number of categories.
""")

code("""# Elbow plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(list(K_RANGE), inertias, 'o-', color='steelblue', linewidth=2)
axes[0].axvline(x=10, color='red', linestyle='--', label='True K=10')
axes[0].set_xlabel('Number of Clusters (K)')
axes[0].set_ylabel('Inertia (Within-Cluster SSE)')
axes[0].set_title('Elbow Method')
axes[0].legend()
axes[0].set_xticks(list(K_RANGE))

axes[1].plot(list(K_RANGE), silhouette_scores, 's-', color='darkorange', linewidth=2)
axes[1].axvline(x=10, color='red', linestyle='--', label='True K=10')
best_sil_k = list(K_RANGE)[int(np.argmax(silhouette_scores))]
axes[1].axvline(x=best_sil_k, color='green', linestyle=':', label=f'Best Silhouette K={best_sil_k}')
axes[1].set_xlabel('Number of Clusters (K)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score vs K')
axes[1].legend()
axes[1].set_xticks(list(K_RANGE))

plt.tight_layout()
plt.savefig('figures/q1c_elbow_silhouette.png', bbox_inches='tight')
plt.show()
""")

code("""# --- Cross-Validation Method (K-Fold silhouette stability) ---
# For each K, split train into folds; fit on train-fold, score silhouette on held-out fold.
N_FOLDS = 5
kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
cv_silhouette = {k: [] for k in K_RANGE}

# Use a subset for CV speed while keeping representativeness
CV_N = 20000
cv_idx = np.random.RandomState(RANDOM_STATE).choice(len(X_train_scaled), CV_N, replace=False)
X_cv = X_train_scaled[cv_idx]

print(f'Cross-validation silhouette (N={CV_N}, {N_FOLDS}-fold)...')
for k in K_RANGE:
    for fold, (tr, va) in enumerate(kf.split(X_cv)):
        mbk = MiniBatchKMeans(n_clusters=k, random_state=RANDOM_STATE,
                              batch_size=1024, n_init=3, max_iter=80)
        mbk.fit(X_cv[tr])
        pred_va = mbk.predict(X_cv[va])
        # Need at least 2 clusters present in validation predictions
        if len(np.unique(pred_va)) > 1:
            sil = silhouette_score(X_cv[va], pred_va)
            cv_silhouette[k].append(sil)
    mean_sil = np.mean(cv_silhouette[k]) if cv_silhouette[k] else float('nan')
    print(f'  K={k:2d} | CV Silhouette mean={mean_sil:.4f} ± {np.std(cv_silhouette[k]):.4f}')

cv_means = [np.mean(cv_silhouette[k]) for k in K_RANGE]
best_cv_k = list(K_RANGE)[int(np.argmax(cv_means))]
""")

md("""Unsupervised criteria often prefer fewer clusters than the true label count because some Fashion MNIST classes (e.g. Shirt vs T-shirt/top, Sneaker vs Ankle boot) are visually similar and merge under pixel-space Euclidean distance. We adopt K equal to the true number of categories for centroid analysis to align with known labels, while noting that CV/silhouette often favour a smaller K.
""")

code("""fig, ax = plt.subplots(figsize=(10, 5))
means = [np.mean(cv_silhouette[k]) for k in K_RANGE]
stds = [np.std(cv_silhouette[k]) for k in K_RANGE]
ax.errorbar(list(K_RANGE), means, yerr=stds, fmt='o-', color='purple',
            capsize=4, linewidth=2, label='CV Silhouette ± std')
ax.axvline(x=10, color='red', linestyle='--', label='True K=10')
ax.axvline(x=best_cv_k, color='green', linestyle=':', label=f'Best CV K={best_cv_k}')
ax.set_xlabel('Number of Clusters (K)')
ax.set_ylabel('Silhouette Score')
ax.set_title('Cross-Validation Method for Selecting K')
ax.set_xticks(list(K_RANGE))
ax.legend()
plt.tight_layout()
plt.savefig('figures/q1c_cv_silhouette.png', bbox_inches='tight')
plt.show()

# Heuristic elbow: largest relative drop slowdown (2nd difference of inertia)
inertia_arr = np.array(inertias)
# Knee via maximum curvature approximation on log-inertia
diffs = np.diff(inertia_arr)
second = np.diff(diffs)
elbow_k = list(K_RANGE)[int(np.argmax(second)) + 1]  # offset for 2nd diff

print('=' * 60)
print('BEST K COMPARISON')
print('=' * 60)
print(f'Elbow Method (approx. knee)     : K = {elbow_k}')
print(f'Silhouette (full subsample)     : K = {best_sil_k}')
print(f'Cross-Validation Silhouette     : K = {best_cv_k}')
print(f'True number of categories       : K = {K_TRUE}')

BEST_K = K_TRUE  # use true categories for interpretable centroid visualization
""")

md("""## 1(d)
""")

md("""Centroids resemble recognizable garment silhouettes (trousers, bags, footwear, coats). Confusion between Shirt/T-shirt/Pullover/Coat is expected because these classes share similar outlines in 28×28 grayscale. Footwear clusters (sandal, sneaker, ankle boot) also partially overlap. Overall, K-Means recovers coarse visual structure even without labels.
""")

code("""# Fit final MiniBatchKMeans with BEST_K
mbk_final = MiniBatchKMeans(
    n_clusters=BEST_K, random_state=RANDOM_STATE,
    batch_size=2048, n_init=10, max_iter=100
)
final_cluster_labels = mbk_final.fit_predict(X_train_scaled)
centroids = mbk_final.cluster_centers_

fig, axes = plt.subplots(2, 5, figsize=(12, 5))
fig.suptitle(f'Cluster Centroids (K={BEST_K})', fontsize=14, fontweight='bold')
for i in range(BEST_K):
    ax = axes[i // 5, i % 5]
    ax.imshow(centroids[i].reshape(28, 28), cmap='gray')
    ax.set_title(f'Centroid {i}', fontsize=10)
    ax.axis('off')
plt.tight_layout()
plt.savefig('figures/q1d_centroids.png', bbox_inches='tight')
plt.show()

# Map each cluster to the majority true label (for interpretability)
print('Cluster → Majority True Class Mapping:')
print('-' * 50)
for c in range(BEST_K):
    mask = final_cluster_labels == c
    majority = Counter(y_train_full[mask]).most_common(3)
    maj_str = ', '.join([f'{FASHION_LABELS[lab]}({cnt})' for lab, cnt in majority])
    print(f'  Cluster {c}: n={mask.sum():5d} | Top labels: {maj_str}')
""")

# ============================================================
# Q2
# ============================================================
md("""# 2.
## 2(a)
""")

code("""# Reuse loaded data; apply MinMaxScaler to obtain normalized covariates X
scaler_pca = MinMaxScaler()
X_train_norm = scaler_pca.fit_transform(X_train_full)  # covariates X for training
X_test_norm = scaler_pca.transform(X_test)

print('Normalized covariates X (training):')
print(f'  Shape : {X_train_norm.shape}')
print(f'  Min   : {X_train_norm.min():.4f}')
print(f'  Max   : {X_train_norm.max():.4f}')
print(f'  Mean  : {X_train_norm.mean():.4f}')
""")

md("""## 2(b)
""")

code("""pca_2d = PCA(n_components=2, random_state=RANDOM_STATE)
X_train_pca = pca_2d.fit_transform(X_train_norm)
X_test_pca = pca_2d.transform(X_test_norm)

print('PCA (n_components=2) fitted on training set')
print(f'  Explained variance ratio : {pca_2d.explained_variance_ratio_}')
print(f'  Cumulative explained var : {pca_2d.explained_variance_ratio_.sum():.4f}')
print(f'  Singular values          : {pca_2d.singular_values_}')
""")

md("""## 2(c)
""")

md("""**Comparison**

- PCA is a linear projection maximizing variance; the first two PCs explain only a modest fraction of total variance, so classes heavily overlap.
- t-SNE is nonlinear and preserves local neighborhoods, producing clearer visual clusters — better for qualitative inspection, but not a global linear embedding.
- K-Means structure is more separable under t-SNE than under 2D PCA.
""")

code("""# PCA projection of the same t-SNE subsample, colored by K-Means clusters and true labels
X_pca_sample = X_train_pca[tsne_idx]

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

sc = axes[0, 0].scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=y_tsne_true,
                        cmap='tab10', s=8, alpha=0.6)
axes[0, 0].set_title('PCA — True Labels')
axes[0, 0].set_xlabel('PC1')
axes[0, 0].set_ylabel('PC2')
plt.colorbar(sc, ax=axes[0, 0], ticks=range(10))

sc = axes[0, 1].scatter(X_pca_sample[:, 0], X_pca_sample[:, 1], c=y_tsne_cluster,
                        cmap='tab10', s=8, alpha=0.6)
axes[0, 1].set_title('PCA — K-Means Clusters')
axes[0, 1].set_xlabel('PC1')
axes[0, 1].set_ylabel('PC2')
plt.colorbar(sc, ax=axes[0, 1], ticks=range(BEST_K))

sc = axes[1, 0].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne_true,
                        cmap='tab10', s=8, alpha=0.6)
axes[1, 0].set_title('t-SNE — True Labels')
axes[1, 0].set_xlabel('t-SNE 1')
axes[1, 0].set_ylabel('t-SNE 2')
plt.colorbar(sc, ax=axes[1, 0], ticks=range(10))

sc = axes[1, 1].scatter(X_embedded[:, 0], X_embedded[:, 1], c=y_tsne_cluster,
                        cmap='tab10', s=8, alpha=0.6)
axes[1, 1].set_title('t-SNE — K-Means Clusters')
axes[1, 1].set_xlabel('t-SNE 1')
axes[1, 1].set_ylabel('t-SNE 2')
plt.colorbar(sc, ax=axes[1, 1], ticks=range(BEST_K))

plt.suptitle('PCA vs t-SNE Visualization of Clustering Results', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q2c_pca_vs_tsne.png', bbox_inches='tight')
plt.show()
""")

# ============================================================
# Q3
# ============================================================
md("""# 3.
## 3(a)
""")

code("""X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_norm, y_train_full,
    test_size=0.2, random_state=RANDOM_STATE, stratify=y_train_full
)

print(f'Train split      : {X_tr.shape[0]:,} samples')
print(f'Validation split : {X_val.shape[0]:,} samples')
print(f'Test split       : {X_test_norm.shape[0]:,} samples')
""")

md("""## 3(b)
""")

code("""alphas = [1e-3, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0]
nb_val_accs = []

print('Tuning MultinomialNB alpha...')
for a in alphas:
    nb = MultinomialNB(alpha=a)
    nb.fit(X_tr, y_tr)
    acc = accuracy_score(y_val, nb.predict(X_val))
    nb_val_accs.append(acc)
    print(f'  alpha={a:<8} | Val Acc={acc:.4f}')

best_alpha = alphas[int(np.argmax(nb_val_accs))]
print(f'\\nBest alpha on validation: {best_alpha}')

# Refit on full train split (tr+val = original training) with best alpha, evaluate on test
nb_best = MultinomialNB(alpha=best_alpha)
nb_best.fit(X_train_norm, y_train_full)
y_test_pred_nb = nb_best.predict(X_test_norm)
nb_test_acc = accuracy_score(y_test, y_test_pred_nb)
print(f'NB Test Accuracy (alpha={best_alpha}): {nb_test_acc:.4f}')
""")

code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(alphas, nb_val_accs, 'o-', color='teal', linewidth=2)
ax.axvline(x=best_alpha, color='red', linestyle='--', label=f'Best α={best_alpha}')
ax.set_xscale('log')
ax.set_xlabel('alpha (α)')
ax.set_ylabel('Validation Accuracy')
ax.set_title('MultinomialNB: Validation Accuracy vs α')
ax.legend()
plt.tight_layout()
plt.savefig('figures/q3b_nb_alpha.png', bbox_inches='tight')
plt.show()
""")

md("""## 3(c)
""")

code("""# Apply PCA model from Q.2 to train/val/test for 2D embedding
X_tr_pca = pca_2d.transform(X_tr)
X_val_pca = pca_2d.transform(X_val)
X_test_pca_emb = pca_2d.transform(X_test_norm)  # same as X_test_pca

# Also need NB predictions in PCA space? No — NB was on full features.
# For KNN we use 2D PCA embedding as required.

neighbor_grid = [1, 3, 5, 7, 9, 11, 15, 21, 31, 51]
knn_val_accs = []

print('Tuning KNN n_neighbors on 2D PCA embedding...')
for k in neighbor_grid:
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_tr_pca, y_tr)
    acc = accuracy_score(y_val, knn.predict(X_val_pca))
    knn_val_accs.append(acc)
    print(f'  n_neighbors={k:<3} | Val Acc={acc:.4f}')

best_n_neighbors = neighbor_grid[int(np.argmax(knn_val_accs))]
print(f'\\nBest n_neighbors: {best_n_neighbors}')

# Refit on full training set PCA embedding
X_train_pca_full = pca_2d.transform(X_train_norm)
knn_best = KNeighborsClassifier(n_neighbors=best_n_neighbors, n_jobs=-1)
knn_best.fit(X_train_pca_full, y_train_full)
y_test_pred_knn = knn_best.predict(X_test_pca_emb)
knn_test_acc = accuracy_score(y_test, y_test_pred_knn)
print(f'KNN Test Accuracy (k={best_n_neighbors}, PCA-2D): {knn_test_acc:.4f}')
""")

code("""fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(neighbor_grid, knn_val_accs, 's-', color='darkred', linewidth=2)
ax.axvline(x=best_n_neighbors, color='red', linestyle='--',
           label=f'Best k={best_n_neighbors}')
ax.set_xlabel('n_neighbors')
ax.set_ylabel('Validation Accuracy')
ax.set_title('KNN on PCA-2D: Validation Accuracy vs n_neighbors')
ax.legend()
plt.tight_layout()
plt.savefig('figures/q3c_knn_k.png', bbox_inches='tight')
plt.show()
""")

md("""## 3(d)
""")

code("""# NB predictions were on original features; for fair 2D plot we still color PCA points
# by NB predicted labels (NB itself is not trained on PCA).
# Optionally also show NB trained on PCA for visualization consistency — assignment asks
# for NB labels on the projected test samples.

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

titles_preds = [
    ('Ground Truth', y_test),
    (f'KNN Predictions (k={best_n_neighbors})', y_test_pred_knn),
    (f'Naïve Bayes Predictions (α={best_alpha})', y_test_pred_nb),
]

for ax, (title, labs) in zip(axes, titles_preds):
    sc = ax.scatter(X_test_pca_emb[:, 0], X_test_pca_emb[:, 1], c=labs,
                    cmap='tab10', s=6, alpha=0.5)
    ax.set_title(title)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    cbar = plt.colorbar(sc, ax=ax, ticks=range(10))
    cbar.ax.set_yticklabels([str(i) for i in range(10)])

plt.suptitle('PCA-Projected Test Samples', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('figures/q3d_pca_predictions.png', bbox_inches='tight')
plt.show()

print(f'Test Accuracy — KNN: {knn_test_acc:.4f} | NB: {nb_test_acc:.4f}')
""")

md("""## 3(e)
""")

code("""def plot_stylish_confusion(y_true, y_pred, title, filename):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(9, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                  display_labels=[FASHION_LABELS[i] for i in range(10)])
    disp.plot(ax=ax, cmap='Blues', colorbar=True, xticks_rotation=45)
    ax.set_title(title, fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(filename, bbox_inches='tight')
    plt.show()
    return cm

cm_knn = plot_stylish_confusion(
    y_test, y_test_pred_knn,
    f'Confusion Matrix — KNN (k={best_n_neighbors}, PCA-2D) | Acc={knn_test_acc:.3f}',
    'figures/q3e_cm_knn.png'
)
cm_nb = plot_stylish_confusion(
    y_test, y_test_pred_nb,
    f'Confusion Matrix — MultinomialNB (α={best_alpha}) | Acc={nb_test_acc:.3f}',
    'figures/q3e_cm_nb.png'
)

print('KNN Confusion Matrix:\\n', cm_knn)
print('\\nNB Confusion Matrix:\\n', cm_nb)
""")

# ============================================================
# Q4
# ============================================================
md("""# 4.
## 4(a)
""")

code("""class FashionMNISTDataset(Dataset):
    \"\"\"Basic PyTorch Dataset loading Fashion MNIST arrays from memory/files.\"\"\"

    def __init__(self, images, labels, transform=None):
        \"\"\"
        Args:
            images: np.ndarray of shape (N, 784) or (N, 28, 28), values in [0, 255] or [0, 1]
            labels: array-like of shape (N,)
            transform: optional torchvision transform applied to PIL Image
        \"\"\"
        self.images = images
        self.labels = np.asarray(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        if img.ndim == 1:
            img = img.reshape(28, 28)
        # Scale to 0-255 uint8 for PIL
        if img.max() <= 1.0:
            img_u8 = (img * 255).astype(np.uint8)
        else:
            img_u8 = img.astype(np.uint8)
        pil = Image.fromarray(img_u8, mode='L')
        if self.transform is not None:
            tensor = self.transform(pil)
        else:
            tensor = transforms.ToTensor()(pil)
        label = int(self.labels[idx])
        return tensor, label


# Also support loading from saved .npz files for the "load data from files" requirement
np.savez('outputs/fashion_mnist_arrays.npz',
         X_train=X_train_full, y_train=y_train_full,
         X_test=X_test, y_test=y_test)
print('Saved arrays to outputs/fashion_mnist_arrays.npz')

data = np.load('outputs/fashion_mnist_arrays.npz')
X_file_train, y_file_train = data['X_train'], data['y_train']
X_file_test, y_file_test = data['X_test'], data['y_test']

basic_transform = transforms.Compose([
    transforms.ToTensor(),  # -> [0,1], shape (1,28,28)
])

# Split training into train/val (80/20 stratified)
tr_idx, val_idx = train_test_split(
    np.arange(len(y_file_train)), test_size=0.2,
    random_state=RANDOM_STATE, stratify=y_file_train
)

train_ds = FashionMNISTDataset(X_file_train[tr_idx], y_file_train[tr_idx], transform=basic_transform)
val_ds = FashionMNISTDataset(X_file_train[val_idx], y_file_train[val_idx], transform=basic_transform)
test_ds = FashionMNISTDataset(X_file_test, y_file_test, transform=basic_transform)

BATCH_SIZE = 128
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Train: {len(train_ds):,} | Val: {len(val_ds):,} | Test: {len(test_ds):,}')
sample_x, sample_y = train_ds[0]
print(f'Sample tensor shape: {sample_x.shape}, label: {sample_y} ({FASHION_LABELS[sample_y]})')
""")

md("""## 4(b)
""")

code("""class BasicCNN(nn.Module):
    \"\"\"3-layer CNN: 2 conv blocks + 1 dense classifier head (dense layers included).\"\"\"

    def __init__(self, num_classes=10):
        super().__init__()
        # Conv layer 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)          # 28 -> 14
        # Conv layer 2
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)          # 14 -> 7
        # Conv layer 3
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)          # 7 -> 3
        # Densely connected layers
        self.fc1 = nn.Linear(128 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


baseline_model = BasicCNN().to(DEVICE)
print(baseline_model)
print('\\nParameter count:', sum(p.numel() for p in baseline_model.parameters() if p.requires_grad))
""")

md("""## 4(c)
""")

code("""def run_epoch(model, loader, criterion, optimizer=None):
    \"\"\"One epoch of train (optimizer set) or eval (optimizer=None).\"\"\"
    is_train = optimizer is not None
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            loss = criterion(outputs, labels)
            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, epochs=5, lr=1e-3, ckpt_path='outputs/best_cnn.pt'):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, optimizer=None)
        history['train_loss'].append(tr_loss)
        history['train_acc'].append(tr_acc)
        history['val_loss'].append(va_loss)
        history['val_acc'].append(va_acc)
        improved = ''
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save({'model_state': model.state_dict(), 'val_acc': va_acc, 'epoch': epoch},
                       ckpt_path)
            improved = ' *best*'
        print(f'Epoch {epoch:02d}/{epochs} | '
              f'train loss={tr_loss:.4f} acc={tr_acc:.4f} | '
              f'val loss={va_loss:.4f} acc={va_acc:.4f} | '
              f'{time.time()-t0:.1f}s{improved}')
    return history, best_val_acc


EPOCHS_BASE = 5
print(f'Training baseline CNN for {EPOCHS_BASE} epochs...')
hist_base, best_val_base = train_model(
    baseline_model, train_loader, val_loader,
    epochs=EPOCHS_BASE, lr=1e-3, ckpt_path='outputs/best_cnn_baseline.pt'
)
print(f'Best validation accuracy (baseline): {best_val_base:.4f}')
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(12, 4))
epochs_axis = range(1, len(hist_base['train_acc']) + 1)

axes[0].plot(epochs_axis, hist_base['train_loss'], 'o-', label='Train')
axes[0].plot(epochs_axis, hist_base['val_loss'], 's-', label='Validation')
axes[0].set_title('Baseline CNN — Loss')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Cross-Entropy Loss')
axes[0].legend()

axes[1].plot(epochs_axis, hist_base['train_acc'], 'o-', label='Train')
axes[1].plot(epochs_axis, hist_base['val_acc'], 's-', label='Validation')
axes[1].set_title('Baseline CNN — Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/q4c_training_curves.png', bbox_inches='tight')
plt.show()
""")

md("""## 4(d)
""")

code("""def evaluate_detailed(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_labels.append(labels.numpy())
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_labels)
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
    }
    return metrics, y_true, y_pred


# Load best baseline checkpoint
ckpt = torch.load('outputs/best_cnn_baseline.pt', map_location=DEVICE)
baseline_model.load_state_dict(ckpt['model_state'])
base_metrics, y_true_cnn, y_pred_cnn = evaluate_detailed(baseline_model, test_loader)

print('Baseline CNN — Test Metrics')
print('-' * 40)
for k, v in base_metrics.items():
    print(f'  {k.capitalize():12s}: {v:.4f}')

plot_stylish_confusion(
    y_true_cnn, y_pred_cnn,
    f'CNN Baseline Confusion Matrix | Acc={base_metrics[\"accuracy\"]:.3f}',
    'figures/q4d_cm_cnn_baseline.png'
)
""")

md("""## 4(e)
""")

md("""**Design choices**

1. **Data augmentation** (random horizontal flip + small rotation) — Fashion items have natural left/right variation; flips help shirts/bags generalize.
2. **Slightly wider network + BatchNorm** — stabilizes training and increases capacity.
3. **Lower learning rate with more epochs** — finer convergence after augmentation noise.
""")

code("""class ImprovedCNN(nn.Module):
    \"\"\"Expanded CNN with BatchNorm for better generalization.\"\"\"

    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.15),

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2), nn.Dropout2d(0.20),

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


aug_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])
eval_transform = transforms.Compose([transforms.ToTensor()])

train_ds_aug = FashionMNISTDataset(X_file_train[tr_idx], y_file_train[tr_idx], transform=aug_transform)
val_ds_imp = FashionMNISTDataset(X_file_train[val_idx], y_file_train[val_idx], transform=eval_transform)
test_ds_imp = FashionMNISTDataset(X_file_test, y_file_test, transform=eval_transform)

train_loader_aug = DataLoader(train_ds_aug, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader_imp = DataLoader(val_ds_imp, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader_imp = DataLoader(test_ds_imp, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

improved_model = ImprovedCNN().to(DEVICE)
print(improved_model)
print('Parameters:', sum(p.numel() for p in improved_model.parameters() if p.requires_grad))

EPOCHS_IMP = 8
print(f'\\nTraining improved CNN for {EPOCHS_IMP} epochs...')
hist_imp, best_val_imp = train_model(
    improved_model, train_loader_aug, val_loader_imp,
    epochs=EPOCHS_IMP, lr=5e-4, ckpt_path='outputs/best_cnn_improved.pt'
)
print(f'Best validation accuracy (improved): {best_val_imp:.4f}')
""")

code("""ckpt_imp = torch.load('outputs/best_cnn_improved.pt', map_location=DEVICE)
improved_model.load_state_dict(ckpt_imp['model_state'])
imp_metrics, y_true_imp, y_pred_imp = evaluate_detailed(improved_model, test_loader_imp)

print('Improved CNN — Test Metrics')
print('-' * 40)
for k, v in imp_metrics.items():
    print(f'  {k.capitalize():12s}: {v:.4f}')

print('\\nComparison vs Baseline:')
cmp = pd.DataFrame({
    'Baseline': base_metrics,
    'Improved': imp_metrics,
    'Delta': {k: imp_metrics[k] - base_metrics[k] for k in base_metrics}
})
display(cmp)

plot_stylish_confusion(
    y_true_imp, y_pred_imp,
    f'Improved CNN Confusion Matrix | Acc={imp_metrics[\"accuracy\"]:.3f}',
    'figures/q4e_cm_cnn_improved.png'
)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
ep = range(1, len(hist_imp['train_acc']) + 1)
axes[0].plot(ep, hist_imp['train_acc'], 'o-', label='Train')
axes[0].plot(ep, hist_imp['val_acc'], 's-', label='Val')
axes[0].set_title('Improved CNN — Accuracy')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Accuracy'); axes[0].legend()
axes[1].plot(ep, hist_imp['train_loss'], 'o-', label='Train')
axes[1].plot(ep, hist_imp['val_loss'], 's-', label='Val')
axes[1].set_title('Improved CNN — Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Loss'); axes[1].legend()
plt.tight_layout()
plt.savefig('figures/q4e_improved_curves.png', bbox_inches='tight')
plt.show()
""")

# ============================================================
# Q5
# ============================================================
md("""# 5.
""")

code("""GUTENBERG_URL = 'https://www.gutenberg.org/browse/scores/top'
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (compatible; STAT7008A-Assignment3/1.0; educational)'
}

print(f'Fetching {GUTENBERG_URL} ...')
resp = requests.get(GUTENBERG_URL, headers=HEADERS, timeout=60)
resp.raise_for_status()
soup = BeautifulSoup(resp.text, 'lxml')

# The "Top 100 EBooks yesterday" (or similar) section lists books as:
#   <ol>
#     <li><a href="/ebooks/1342">Pride and Prejudice by Jane Austen (12345)</a></li>
#   </ol>
books_data = []

# Find header that mentions Top 100 EBooks
candidate_ols = []
for header in soup.find_all(['h2', 'h3']):
    text = header.get_text(' ', strip=True).lower()
    if 'top 100' in text and 'ebook' in text:
        ol = header.find_next('ol')
        if ol:
            candidate_ols.append((header.get_text(' ', strip=True), ol))

if not candidate_ols:
    # Fallback: first ordered list on the page with ebook links
    for ol in soup.find_all('ol'):
        links = ol.find_all('a', href=re.compile(r'/ebooks/\\d+'))
        if len(links) >= 20:
            candidate_ols.append(('fallback-ol', ol))
            break

assert candidate_ols, 'Could not locate Top 100 ebook list on Gutenberg page'
section_name, top_ol = candidate_ols[0]
print(f'Using section: {section_name}')

items = top_ol.find_all('li', recursive=False)[:20]
for rank, li in enumerate(items, start=1):
    a = li.find('a', href=re.compile(r'/ebooks/\\d+'))
    if a is None:
        continue
    href = a.get('href', '')
    book_id = re.search(r'/ebooks/(\\d+)', href).group(1)
    raw = a.get_text(' ', strip=True)
    # Typical format: "Title by Author (downloads)" or "Title (downloads)"
    downloads = None
    m_dl = re.search(r'\\(([0-9,]+)\\)\\s*$', raw)
    if m_dl:
        downloads = int(m_dl.group(1).replace(',', ''))
        raw_main = raw[:m_dl.start()].strip()
    else:
        raw_main = raw

    # Split title / author on last " by "
    if ' by ' in raw_main:
        title, author = raw_main.rsplit(' by ', 1)
        title, author = title.strip(), author.strip()
    else:
        title, author = raw_main, 'Unknown'

    books_data.append({
        'rank': rank,
        'title': title,
        'author': author,
        'downloads': downloads,
        'book_id': book_id,
        'url': f'https://www.gutenberg.org/ebooks/{book_id}'
    })

gutenberg_df = pd.DataFrame(books_data)
csv_path = 'outputs/gutenberg_top20.csv'
gutenberg_df.to_csv(csv_path, index=False)
print(f'Saved {len(gutenberg_df)} books to {csv_path}')
print()
print('Project Gutenberg — Top 20 Books')
print('=' * 90)
display(gutenberg_df)
print(gutenberg_df.to_string(index=False))
""")

# ============================================================
# Q6
# ============================================================
md("""# 6.
""")

code("""BASE_URL = 'http://books.toscrape.com'
catalogue_urls = [
    f'{BASE_URL}/catalogue/page-{i}.html' for i in range(1, 4)
]
# page-1 also available as index; catalogue/page-1.html works on books.toscrape.com

RATING_MAP = {'One': 1, 'Two': 2, 'Three': 3, 'Four': 4, 'Five': 5}

scraped_books = []
for page_num, url in enumerate(catalogue_urls, start=1):
    print(f'Scraping page {page_num}: {url}')
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    page = BeautifulSoup(r.text, 'lxml')
    articles = page.select('article.product_pod')
    for art in articles:
        title = art.h3.a.get('title') or art.h3.a.get_text(strip=True)
        price_text = art.select_one('p.price_color').get_text(strip=True)
        # price like "£51.77" — strip currency
        price = float(re.sub(r'[^0-9.]', '', price_text))
        rating_cls = [c for c in art.select_one('p.star-rating')['class'] if c != 'star-rating'][0]
        rating = RATING_MAP[rating_cls]
        availability = art.select_one('p.instock.availability').get_text(strip=True)
        scraped_books.append({
            'title': title,
            'price': price,
            'rating': rating,
            'rating_label': rating_cls,
            'availability': availability,
            'page': page_num
        })

books_df = pd.DataFrame(scraped_books)
books_df.to_csv('outputs/books_to_scrape.csv', index=False)
print(f'\\nScraped {len(books_df)} books from 3 pages')
display(books_df.head(10))
""")

md("""## 6(b)
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(books_df['price'], bins=15, color='steelblue', edgecolor='black')
axes[0].set_title('Distribution of Book Prices')
axes[0].set_xlabel('Price (£)')
axes[0].set_ylabel('Frequency')

sns.boxplot(y=books_df['price'], ax=axes[1], color='lightblue')
axes[1].set_title('Book Price Boxplot')
axes[1].set_ylabel('Price (£)')

plt.tight_layout()
plt.savefig('figures/q6b_price_distribution.png', bbox_inches='tight')
plt.show()

print(books_df['price'].describe())
""")

md("""## 6(c)
""")

code("""avg_rating = books_df['rating'].mean()
print(f'Average rating for scraped books: {avg_rating:.3f} / 5')
print(f'(Based on {len(books_df)} books)')
""")

md("""## 6(d)
""")

code("""rating_counts = books_df['rating'].value_counts().sort_index()
rating_summary = pd.DataFrame({
    'Stars': rating_counts.index,
    'Count': rating_counts.values,
    'Percentage': (rating_counts.values / rating_counts.sum() * 100).round(1)
})
display(rating_summary)

fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(rating_counts.index.astype(str), rating_counts.values, color='coral', edgecolor='black')
ax.set_xlabel('Star Rating')
ax.set_ylabel('Number of Books')
ax.set_title('Books in Each Rating Category')
for i, v in enumerate(rating_counts.values):
    ax.text(i, v + 0.3, str(v), ha='center')
plt.tight_layout()
plt.savefig('figures/q6d_rating_counts.png', bbox_inches='tight')
plt.show()
""")

md("""## 6(e)
""")

code("""fig, axes = plt.subplots(1, 2, figsize=(13, 4))

sns.boxplot(x='rating', y='price', data=books_df, ax=axes[0], palette='Set2')
axes[0].set_title('Price by Rating')
axes[0].set_xlabel('Star Rating')
axes[0].set_ylabel('Price (£)')

# Jittered scatter + mean line
rng = np.random.RandomState(RANDOM_STATE)
jitter = rng.uniform(-0.15, 0.15, size=len(books_df))
axes[1].scatter(books_df['rating'] + jitter, books_df['price'], alpha=0.6, s=40)
means = books_df.groupby('rating')['price'].mean()
axes[1].plot(means.index, means.values, 'r-o', label='Mean price')
axes[1].set_xlabel('Star Rating')
axes[1].set_ylabel('Price (£)')
axes[1].set_title('Price vs Rating')
axes[1].legend()

plt.tight_layout()
plt.savefig('figures/q6e_price_vs_rating.png', bbox_inches='tight')
plt.show()

corr = books_df['price'].corr(books_df['rating'])
print(f'Pearson correlation (price, rating): {corr:.4f}')
print(books_df.groupby('rating')['price'].agg(['count', 'mean', 'median', 'std']).round(2))
""")

md("""## 6(f)
""")

md("""**Key findings**

1. Prices on the first three catalogue pages span a wide range with a roughly unimodal distribution centred near the mean price.
2. Ratings are discrete (1–5); the modal category is reported in the numeric summary below.
3. The Pearson correlation between price and rating is weak in magnitude, suggesting little linear relationship between how expensive a book is and its star rating on this practice site.
4. All listed books on these pages report availability as "In stock".
""")

code("""print('=' * 70)
print('BOOKS TO SCRAPE — SUMMARY REPORT')
print('=' * 70)
print(f'Pages scraped              : 3')
print(f'Total books                : {len(books_df)}')
print(f'Price range                : £{books_df[\"price\"].min():.2f} – £{books_df[\"price\"].max():.2f}')
print(f'Mean / median price        : £{books_df[\"price\"].mean():.2f} / £{books_df[\"price\"].median():.2f}')
print(f'Average star rating        : {avg_rating:.3f} / 5')
print(f'Most common rating         : {rating_counts.idxmax()} stars ({rating_counts.max()} books)')
print(f'Price–rating correlation   : {corr:.4f}')
print('=' * 70)
""")

# ============================================================
# Q7
# ============================================================
md("""# 7.
""")

code("""class FeatureVisualizer:
    def __init__(self):
        # Hint 1: Initialize model
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.model.eval()

        # Hint 2: Set up feature storage
        self.features = {}

        # Hint 3: Register hooks
        self.model.conv1.register_forward_hook(self.get_features('conv1'))
        self.model.layer2.register_forward_hook(self.get_features('layer2'))
        self.model.layer4.register_forward_hook(self.get_features('layer4'))

        # Hint 4: Set up image transforms
        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])

        # Hint 5: Layer names for display
        self.layer_names = {
            'conv1': 'First Layer',
            'layer2': 'Middle Layer',
            'layer4': 'Final Layer'
        }

    def get_features(self, layer_name):
        # Hint 6: Create hook function
        def hook(module, input, output):
            # Store the output features; detach to prevent memory leaks
            self.features[layer_name] = output.detach()
        return hook

    def load_image(self, url):
        # Hint 7: Load and process image
        try:
            headers = {
                'User-Agent': (
                    'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
                    'AppleWebKit/537.36 (KHTML, like Gecko) '
                    'Chrome/120.0.0.0 Safari/537.36'
                )
            }
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert('RGB')
            # Cache locally for reproducibility
            image.save('data/cat03.jpg')
            tensor = self.transform(image)
            return tensor.unsqueeze(0)  # Add batch dimension
        except Exception as e:
            print(f'Error loading image: {str(e)}')
            # Fallback to local cache if download is blocked
            try:
                image = Image.open('data/cat03.jpg').convert('RGB')
                print('Loaded local fallback image: data/cat03.jpg')
                tensor = self.transform(image)
                return tensor.unsqueeze(0)
            except Exception as e2:
                print(f'Fallback also failed: {e2}')
                return None

    def show_features(self, layer_name):
        # Hint 8: Create visualization
        if layer_name not in self.features:
            return

        feat = self.features[layer_name][0]  # remove batch dim -> (C, H, W)
        n_channels = min(16, feat.shape[0])
        maps = feat[:n_channels].cpu().numpy()

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        for i in range(16):
            ax = axes[i // 4, i % 4]
            if i < n_channels:
                im = ax.imshow(maps[i], cmap='viridis')
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                ax.set_title(f'Ch {i}', fontsize=9)
            ax.axis('off')

        fig.suptitle(f'ResNet-18 Feature Maps — {self.layer_names[layer_name]} ({layer_name})',
                     fontsize=14, fontweight='bold')
        plt.tight_layout()
        out_path = f'figures/q7_features_{layer_name}.png'
        plt.savefig(out_path, bbox_inches='tight')
        plt.show()
        print(f'Saved {out_path}')

        # Hint 9: Print statistics
        print(f'Statistics for {layer_name} ({self.layer_names[layer_name]}):')
        print(f'  Shape : {tuple(self.features[layer_name].shape)}')
        print(f'  Mean  : {feat.mean().item():.6f}')
        print(f'  Std   : {feat.std().item():.6f}')
        print(f'  Min   : {feat.min().item():.6f}')
        print(f'  Max   : {feat.max().item():.6f}')
        print()

    def process_image(self, url):
        # Hint 10: Main processing function
        self.features.clear()
        img = self.load_image(url)
        if img is None:
            return
        with torch.no_grad():
            _ = self.model(img)
        for layer in ['conv1', 'layer2', 'layer4']:
            self.show_features(layer)


def main():
    url = 'https://upload.wikimedia.org/wikipedia/commons/3/3a/Cat03.jpg'
    visualizer = FeatureVisualizer()
    visualizer.process_image(url)


main()
""")

nb.cells = cells
nb.metadata = {
    'kernelspec': {
        'display_name': 'Python 3',
        'language': 'python',
        'name': 'python3'
    },
    'language_info': {
        'name': 'python',
        'version': '3.12.3'
    }
}

out_path = 'STAT7008A_Assignment3.ipynb'
with open(out_path, 'w') as f:
    nbf.write(nb, f)
print(f'Wrote {out_path} with {len(cells)} cells')
