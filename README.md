# STAT7008A Assignment 3 — Machine Learning

Programming for Data Science (The University of Hong Kong)

## Contents

| # | Topic |
|---|--------|
| 1 | K-Means Clustering on Fashion MNIST (EDA, MiniBatchKMeans, Elbow/CV, centroids) |
| 2 | PCA dimension reduction vs t-SNE |
| 3 | MultinomialNB & KNN classification |
| 4 | CNN with PyTorch (baseline + improved) |
| 5 | Project Gutenberg Top 20 scraper |
| 6 | Books to Scrape (3 pages) + analysis |
| 7 | ResNet-18 feature map visualization |

## Files

- `STAT7008A_Assignment3.ipynb` — main submission notebook
- `STAT7008A_Assignment3.html` — HTML export
- `requirements.txt` — Python dependencies
- `figures/` — generated plots
- `outputs/` — CSV results and model checkpoints

## Setup

```bash
pip install -r requirements.txt
python -m jupyter nbconvert --to notebook --execute STAT7008A_Assignment3.ipynb
```

Requires Python >= 3.6 (developed with 3.12).
