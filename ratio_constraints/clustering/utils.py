import numpy as np
from sklearn.cluster import AgglomerativeClustering
from scipy.linalg import inv, eigh
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.covariance import OAS

def test_cluster_family_sizes(cluster_family_sizes, n_clusters, n_total):
    # make sure that the sizes are int
    cluster_family_sizes = {k: int(v) for k, v in cluster_family_sizes.items()}

    # make sure that there is no intersection between families
    assert len(cluster_family_sizes) == 1 or len(set.intersection(*map(set, cluster_family_sizes.keys()))) == 0, "There is an intersection between cluster families"

    # make sure that the clusters 
    clusters_in_families = sum(map(list, cluster_family_sizes.keys()), [])
    assert max(clusters_in_families) < n_clusters and min(clusters_in_families) >= 0, "Unrecognized cluster id in cluster families"

    total_family_size = sum(cluster_family_sizes.values())
    # either we cover all the clusters and data or we leave some cluster and we leave some data
    assert (len(clusters_in_families) == n_clusters and total_family_size == n_total) or (len(clusters_in_families) < n_clusters and total_family_size < n_total), "Inconsistent data distribution on cluster families"

    return cluster_family_sizes


def scale(LM, W):
    if isinstance(LM, list):
        return [M * W for M in LM]
    return LM * W


def m_step(X, A, n_clusters):
    # we can either concatenate everything or use running statistics 🏃

    X = np.concatenate(X, axis=0)
    A = np.concatenate(A, axis=0)

    # compressed assignation matrix
    if len(A.shape) == 1:
        C = np.empty((n_clusters, X.shape[1]), dtype=float)
        for i in range(n_clusters):
            C[i] = X[A == i].mean(axis=0)

    else:
        C = (A.T @ X) / A.sum(axis=0, keepdims=True).T

    return C


def initialize_centroids(X, y, cluster2tag, tag2clusters, init, n_clusters, random_state=None):
    if init == 'k-means++':
        # initiate centroids with sklearn method
        # return kmeans_plusplus(X, n_clusters, random_state=random_state)[0]
        n, n_features = X.shape

        rng = np.random.default_rng(seed=random_state)
        
        if (y == 0).any():
            cluster_centers = rng.choice(X[y == cluster2tag[0]])[np.newaxis, :]
        else:
            cluster_centers = rng.choice(X[y == -1])[np.newaxis, :]

        distances = np.zeros((n, 0), dtype=float)

        while cluster_centers.shape[0] < n_clusters:
            distances = np.concatenate(
                (
                    distances, 
                    ((X[:, np.newaxis, :] - cluster_centers[np.newaxis, -1:, :])**2).sum(axis=2)
                ),
                axis=1
            )
            shortest_distance_to_centroid = distances.min(axis=1)

            cluster_id = cluster_centers.shape[0]

            if (y == cluster_id).any():
                probabilities = (
                    shortest_distance_to_centroid[y == cluster2tag[cluster_id]] / 
                    shortest_distance_to_centroid[y == cluster2tag[cluster_id]].sum(axis=0, keepdims=True)
                ).reshape(-1)
                cluster_centers = np.concatenate((cluster_centers, rng.choice(X[y == cluster2tag[cluster_id]], p=probabilities)[np.newaxis, :]), axis=0)
            else:
                probabilities = (
                    shortest_distance_to_centroid[y == -1] /
                    shortest_distance_to_centroid[y == -1].sum(axis=0, keepdims=True)
                ).reshape(-1)
                cluster_centers = np.concatenate((cluster_centers, rng.choice(X[y == -1], p=probabilities)[np.newaxis, :]), axis=0)

        return cluster_centers
    if init == 'means':
        """
        for each tag, we first take the mean, then take the farthest point from the existing clusters of that tag
        """
        n, n_features = X.shape
        cluster_centers = np.empty((n_clusters, n_features), dtype=float)

        for tag, cluster_ids in tag2clusters.items():
            first_cluster_id = cluster_ids[0]

            # the first center is the mean
            cluster_center = X[y == tag].mean(axis=0)
            cluster_centers[first_cluster_id] = cluster_center

            # then we take the farthest points from existing centers
            distances = np.zeros((n, 0), dtype=float)
            for cluster_id in cluster_ids[1:]:
                distances = np.concatenate(
                    (
                        distances, 
                        ((X[:, np.newaxis, :] - cluster_center[np.newaxis, :])**2).sum(axis=2)
                    ),
                    axis=1
                )
                shortest_distance_to_centroid = distances.min(axis=1)
                shortest_distance_to_centroid[y != tag] = - np.inf

                cluster_center = X[shortest_distance_to_centroid.argmax()]
                cluster_centers[cluster_id] = cluster_center
            
        return cluster_centers

    if init in ["hierarchical-ward", "hierarchical-complete", "hierarchical-average", "hierarchical-single"]:
        """initialize centroids of the same tag using heirarchical clustering"""

        linkage = init.split('-')[1]

        n, n_features = X.shape
        cluster_centers = np.empty((n_clusters, n_features), dtype=float)
        for tag, cluster_ids in tag2clusters.items():
            X_tag = X[y == tag]
            if len(cluster_ids) > 1:
                labels = AgglomerativeClustering(n_clusters=len(cluster_ids), linkage=linkage).fit(X_tag).labels_
                for i in range(len(cluster_ids)):
                    cluster_centers[cluster_ids[i]] = X_tag[labels == i].mean(axis=0)
            else:
                cluster_centers[cluster_ids[0]] = X_tag.mean(axis=0)

        return cluster_centers

    if init == 'random':
        # return X[np.random.choice(X.shape[0], n_clusters, replace=False)]
        
        rng = np.random.default_rng(seed=random_state)

        wo_reference = [k for k in range(n_clusters) if not (y == k).any()]

        cluster_centers = np.zeros((n_clusters, X.shape[1]), dtype=float)

        if len(wo_reference) > 0:
            cluster_centers[wo_reference] = rng.choice(X[y == -1], size=len(wo_reference))

        for k in range(n_clusters):
            if k in wo_reference:
                continue
            cluster_centers[k] = rng.choice(X[y == k])

        return cluster_centers

    if type(init) == np.ndarray:
        return init

    raise ValueError()


def shrunk_cov(covariance, n_samples, n_features):
    """
    OAS
    https://github.com/scikit-learn/scikit-learn/blob/70fdc843a4b8182d97a3508c1a426acc5e87e980/sklearn/covariance/_shrunk_covariance.py#L48
    """

    alpha = np.mean(covariance**2)
    mu = np.trace(covariance) / n_features
    mu_squared = mu**2
    
    num = alpha + mu_squared
    den = (n_samples + 1) * (alpha - mu_squared / n_features)
    shrinkage_ = 1.0 if den == 0 else min(num / den, 1.0)
    
    shrunk_cov = (1.0 - shrinkage_) * covariance
    shrunk_cov.flat[:: n_features + 1] += shrinkage_ * mu
    return shrunk_cov 

def SW(X, C, A):
    _, n_features = X.shape
    covariance = np.zeros((n_features, n_features), dtype=float)

    priors = A.sum(axis=0)
    priors = priors / priors.sum()

    for j in range(A.shape[1]):
        n_samples = A[:, j].sum()

        Xc = X - C[j:j+1, :]

        cluster_covar = ((Xc * A[:, j:j+1]).T @ Xc) / n_samples

        covariance += priors[j] * shrunk_cov(cluster_covar, n_samples, n_features)

    return covariance

def SB(X, C, A):
    out = np.zeros((X.shape[1], X.shape[1]))
    
    X_mean = X.mean(axis=0)
    C = C - X_mean
    
    for j in range(C.shape[0]):
        out += np.outer(C[j], C[j])

    return out

def ST(X):
    n_samples, n_features = X.shape

    X_mean = X.mean(axis=0)
    X = X - X_mean

    covariance = (X.T @ X) / X.shape[0]

    return shrunk_cov(covariance, n_samples, n_features)

def fda_step(X, A, C, n_components):
    Sw = SW(X, C, A)
    St = ST(X)
    Sb = St - Sw

    evals, evecs = eigh(Sb, Sw)
    U = evecs[:, -n_components:].T
    return U
