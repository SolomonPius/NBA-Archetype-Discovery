# NBA Archetype Discovery

Unsupervised discovery of NBA player archetypes. A variational autoencoder (VAE) compresses each player's season stats into a low-dimensional embedding, and k-means clusters the embeddings into archetypes, with no position labels or hand-made categories.

[Open in Google Colab](https://colab.research.google.com/drive/1wf8XbC6_9_wheWtT8hdmzINYOZNP6cji?usp=sharing)

## Results

| | Silhouette score (k = 9) |
|---|---|
| PCA baseline (6 components) | 0.188 |
| **VAE embedding (6-D)** | **0.358** |

The VAE embedding nearly doubles cluster separation compared with a PCA baseline of the same dimension.

## Pipeline

```
21 stats per player  →  standardize  →  VAE encoder  →  6-D embedding (μ)  →  k-means (k = 9)  →  archetypes
```

1. **Data:** 21 per-player stats (shot-location distribution, assisted-shot rates, TS%, 3PAr, FTr, rebound/assist/steal/block/turnover/usage rates, minutes, FGA). Players under 4 minutes per game are removed, leaving 500 players.
2. **Preprocessing:** missing values filled with column medians, then standardized.
3. **Model:** VAE with a 64 → 32 encoder (BatchNorm, ReLU, dropout 0.1), a 6-D latent space and a mirrored decoder. The loss is reconstruction MSE plus a KL term weighted by β = 0.5, so reconstruction stays dominant. It trains with Adam (lr 1e-3, weight decay 1e-4) for 150 epochs. The deterministic mean μ is used as the embedding.
4. **Choosing k:** a silhouette sweep over k = 4–15 (50 k-means restarts each). The highest score was k = 7 (0.375). I chose k = 9 (0.358) because it separates guard roles that k = 7 merges, at a small cost in score.
5. **Interpretation:** each cluster's representative player is the one closest to its centroid in embedding space.

## Discovered archetypes

Labels are my interpretation of each cluster's average stats.

| Cluster | Archetype | Size | Players closest to centroid |
|---|---|---|---|
| 0 | High-usage primary creators | 44 | De'Aaron Fox, Donovan Mitchell, Kevin Porter Jr. |
| 1 | Versatile forwards | 67 | Santi Aldama, Onyeka Okongwu, Tari Eason |
| 2 | Low-usage spot-up shooters | 76 | Eric Gordon, Dorian Finney-Smith, Baylor Scheierman |
| 3 | Rim-running, shot-blocking bigs | 36 | Khaman Maluach, Mark Williams, Dwight Powell |
| 4 | Bench ball-handlers | 50 | Tre Mann, Jordan Poole, Tyler Kolek |
| 5 | Big playmakers / point forwards | 42 | Josh Giddey, Franz Wagner, Julius Randle |
| 6 | Low-minute energy wings and forwards | 74 | Johnny Furphy, Jabari Walker, Jalen Smith |
| 7 | Rebounding interior bigs | 41 | Jakob Poeltl, Daniel Gafford, Nick Richards |
| 8 | 3-and-D role wings | 70 | Toumani Camara, Kevin Huerter, Patrick Williams |

Some sanity checks: LeBron James and Nikola Jokić both land in cluster 5 (big playmakers), and Stephen Curry lands in cluster 0 (primary creators).

## Generalization test

The trained scaler, encoder and k-means model are applied unchanged to the 2015–16 season (`NBA_2016.xlsx`, players with 10+ minutes per game). Assignments are consistent with known roles: for example, Clint Capela and Tyson Chandler go to rim-running bigs, and Kobe Bryant, Jimmy Butler and Stephen Curry go to primary creators.

## Notebook features

- Look up any player's cluster by name
- List every player in a cluster, sorted by any stat
- Predict the archetype of a new player from a stat dictionary

## Files

| File | Contents |
|---|---|
| `Final_iteration.ipynb` | Final VAE pipeline |
| `Iteration_1.ipynb` | First version: standard autoencoder (8-D latent) with fixed k = 10 |
| `NBA_data.xlsx` | Training data, 521 players |
| `NBA_2016.xlsx` | 2015–16 season data for the generalization test |

## Tech

Python, PyTorch, scikit-learn, pandas
