"""Solution's template for user."""
from itertools import combinations
import numpy as np
from collections import defaultdict


def knn_uniqueness(embeddings: np.ndarray, num_neighbors: int) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on knn.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    num_neighbors: int :
        number of neighbors to estimate uniqueness

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    pair_sims = {}
    for pair in combinations(embeddings, 2):
        score = np.linalg.norm(
            pair[0] - pair[1]
        )
        idx_1 = np.where(embeddings == pair[0])[0][0]
        idx_2 = np.where(embeddings == pair[1])[0][0]
        pair_sims[(idx_1, idx_2)] = float(score)

    knn_dict = defaultdict(list)
    for pair in pair_sims:
        knn_dict[pair[0]].append((pair[1], pair_sims[pair]))
        knn_dict[pair[1]].append((pair[0], pair_sims[pair]))

    for key, value in knn_dict.items():
        knn_dict[key] = sorted(value, key=lambda x: x[1])[:num_neighbors]

    uniqueness = []
    for value in knn_dict.values():
        distances = []
        for element in value:
            distances.append(element[1])
        uniqueness.append(np.mean(distances))

    return np.array(uniqueness)
