from typing import Tuple

import numpy as np
from sklearn.neighbors import KernelDensity

DIVERSITY_THRESHOLD = 10

embeddings = {
    1: np.random.normal(size=(4)),
    2: np.random.normal(size=(4)),
    3: np.random.normal(size=(4)),
    4: np.random.normal(size=(4)),
    5: np.random.normal(size=(4))
}


def uniqueness(item_ids: str) -> dict:
    """Calculate uniqueness of each product"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    item_uniqueness = {item_id: 0.0 for item_id in item_ids}

    # Calculate uniqueness
    embeddings_lst = np.array([embeddings[item_id]  for item_id in item_ids])
    uniquenesses = kde_uniqueness(embeddings_lst)
    for item_id, uniqueness in zip(item_ids, uniquenesses):
        item_uniqueness[item_id] = uniqueness

    return item_uniqueness


def diversity(item_ids: str) -> dict:
    """Calculate diversity of group of products"""

    # Parse item IDs
    item_ids = [int(item) for item in item_ids.split(",")]

    # Default answer
    response = {"diversity": 0.0, "reject": True}

    # Calculate diversity
    embeddings_lst = np.array([embeddings[item_id] for item_id in item_ids])
    reject, diversity = group_diversity(embeddings_lst, DIVERSITY_THRESHOLD)

    response["diversity"] = diversity
    response["reject"] = reject

    return response


def kde_uniqueness(embeddings: np.ndarray) -> np.ndarray:
    """Estimate uniqueness of each item in item embeddings group. Based on KDE.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group

    Returns
    -------
    np.ndarray
        uniqueness estimates

    """
    # Fit a kernel density estimator to the item embedding space
    kde = KernelDensity(kernel='gaussian')
    kde.fit(embeddings)

    estimation = 1 / np.exp(kde.score_samples(embeddings))

    return estimation


def group_diversity(embeddings: np.ndarray, threshold: float) -> Tuple[bool, float]:
    """Calculate group diversity based on kde uniqueness.

    Parameters
    ----------
    embeddings: np.ndarray :
        embeddings group
    threshold: float :
       group deversity threshold for reject group

    Returns
    -------
    Tuple[bool, float]
        reject
        group diverstity

    """
    uniquenesses = kde_uniqueness(embeddings)
    diversity = np.mean(uniquenesses)

    decision = (diversity < threshold, diversity)
    return decision
