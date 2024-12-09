import numpy as np
from ortools.graph.python import min_cost_flow
from sklearn.metrics.pairwise import euclidean_distances
from scipy.optimize import linear_sum_assignment
from scipy.special import log_softmax, softmax, logsumexp

def soft_thresholding(x, c):
    return np.sign(x) * np.clip(np.abs(x) - c, a_min=0, a_max=None)


def normalized_soft_thresholding(x, c):
    threshold_x = soft_thresholding(x, c)
    return threshold_x / np.linalg.norm(threshold_x)

# source: https://gist.github.com/mblondel/6f3b7aaad90606b98f71#file-projection_simplex-py-L19
def projection_simplex_sort(v, z=1):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - z
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w


def unconstrained_hard_e_step(X, C, mask): 
    A = []
    for X_, mask_ in zip(X, mask):
        d = euclidean_distances(X_, C, squared=True)
        d[~mask_] = np.inf

        A_ = d.argmin(axis=1)
        A.append(A_)

    return A

def unconstrained_soft_e_step(X, C, mask):
    A = []
    for X_, mask_ in zip(X, mask):
        A_ = - euclidean_distances(X_, C, squared=True)
        A_[~mask_] = - np.inf

        A_ = softmax(A_, axis=1)
        A.append(A_)
    return A

def constrained_hard_e_step(X, C, mask, cluster_family_sizes):
    assert len(cluster_family_sizes) == 1, "for now, only a single size constraint on the O clusters is allowed"

    n_clusters, dim = C.shape

    X_all = np.concatenate(X, axis=0)
    mask_all = np.concatenate(mask, axis=0)
    A_ = np.full((X_all.shape[0],), -1, dtype=int)

    o_clusters, o_clusters_size = list(cluster_family_sizes.items())[0]

    o_clusters = np.array(o_clusters)
    non_o_clusters = np.array([i for i in range(n_clusters) if i not in o_clusters])

    o_clusters_mask = np.full((n_clusters,), False)
    o_clusters_mask[o_clusters] = True

    # calculate the distances
    distances = euclidean_distances(X_all, C, squared=True)
    distances[~mask_all] = np.inf

    # calculate the distance to the closest O cluster and the closest non O cluster
    o_clusters_dist = distances[:, o_clusters_mask].min(axis=1)
    non_o_clusters_dist = distances[:, ~o_clusters_mask].min(axis=1)

    # take the o_cluster_size closest points to O clusters
    o_clusters_X_ids = np.argpartition(o_clusters_dist - non_o_clusters_dist, o_clusters_size).flatten()[:o_clusters_size]
    # and assign them their closest O cluster
    A_[o_clusters_X_ids] = o_clusters[distances[o_clusters_X_ids][:, o_clusters_mask].argmin(axis=1)]
    
    # assign the rest of the points to their closest cluster
    A_[A_ == -1] = non_o_clusters[distances[A_ == -1][:, ~o_clusters_mask].argmin(axis=1)]

    A = []
    start = 0
    for X_ in X:
        A.append(A_[start: start + X_.shape[0]])
        start += X_.shape[0]

    return A


def constrained_hard_e_step_min_cost_flow(X, C, lower_bounds, upper_bounds, mask, compressed=False, integer_rounding_mult=1000):
    """"
    Solve the E step of the constrained k-means problem using min cost flow.

    Parameters
    ----------
    X : numpy.ndarray
        The data points. (num_datapoints, dim)
    C : numpy.ndarray
        The centroids. (num_clusters, dim)
    lower_bounds : list
        The lower bound of each cluster. (num_clusters,)
    upper_bounds : list
        The upper bound of each cluster. (num_clusters,)
    mask : numpy.ndarray
        Mask for unwanted cluster assignments. (num_datapoints, num_clusters)
    compressed : bool
        If True, will return a vector of integers instead of a matrix of booleans
    integer_rounding_mult : int
        Integer multiplier for the costs (default 1000)
    """
    assert C.shape[1] == X.shape[1]
    assert len(lower_bounds) == C.shape[0]
    assert len(upper_bounds) == C.shape[0]

    num_datapoints = X.shape[0]
    n_clusters = C.shape[0]

    dist = euclidean_distances(X, C, squared=True)

    # X_i to C_j
    start_nodes_a = np.arange(num_datapoints).repeat(n_clusters)
    end_nodes_a = np.tile(num_datapoints + np.arange(n_clusters), num_datapoints)
    unit_costs_a = (integer_rounding_mult * dist.reshape(-1)).astype(np.int_)
    capacities_a = np.ones_like(start_nodes_a)

    # C_j to sink (if needed)
    # upper bounds are capacities on the output arc
    start_nodes_b = list()
    end_nodes_b = list()
    capacities_b =list()
    extra_capacity = num_datapoints - lower_bounds.sum()

    for i in range(n_clusters):
        if lower_bounds[i] == upper_bounds[i]:
            # if constraint is an equality,
            # we do not add an output arc,
            # so nothing to do!
            pass
        elif upper_bounds[i] < 0:
            # no upper bound,
            # so max capacity on output arc
            start_nodes_b.append(num_datapoints + i)
            end_nodes_b.append(num_datapoints + n_clusters)
            capacities_b.append(extra_capacity)
        else:
            assert lower_bounds[i] < upper_bounds[i]
            start_nodes_b.append(num_datapoints + i)
            end_nodes_b.append(num_datapoints + n_clusters)
            capacities_b.append(upper_bounds[i] - lower_bounds[i])

    if len(start_nodes_b) > 0:
        start_nodes = np.concatenate([start_nodes_a, start_nodes_b])
        end_nodes = np.concatenate([end_nodes_a, end_nodes_b])
        capacities = np.concatenate([capacities_a, capacities_b])
        unit_costs = np.concatenate([unit_costs_a, np.zeros(len(start_nodes_b), dtype=np.int_)])

        supplies = np.concatenate([
            np.ones(num_datapoints, dtype=np.int_),
            -lower_bounds,
            np.array([-num_datapoints + lower_bounds.sum()], dtype=np.int_)
        ])
    else:
        start_nodes = start_nodes_a
        end_nodes = end_nodes_a
        capacities = capacities_a
        unit_costs = unit_costs_a

        supplies = np.concatenate([
            np.ones(num_datapoints, dtype=np.int_),
            -lower_bounds
        ])

    capacities[~mask.flatten()] = 0
    smcf = min_cost_flow.SimpleMinCostFlow()

    all_arcs = smcf.add_arcs_with_capacity_and_unit_cost(
        start_nodes, end_nodes, capacities, unit_costs
    )

    smcf.set_nodes_supplies(np.arange(0, len(supplies)), supplies)
    status = smcf.solve()

    if status != smcf.OPTIMAL:
        raise RuntimeError("Could not solve the E step. Maybe check the upper/lower bounds?")

    A = smcf.flows(all_arcs[:num_datapoints * n_clusters]).reshape(num_datapoints, n_clusters)
    if compressed:
        A = A.argmax(axis=1)
    return A

def constrained_hard_e_step_linear_sum_assignment(X, C, cluster_sizes, compressed=False):
    """
    Solve the E step of the constrained k-means problem using linear sum assignment.
    This is possible when we have only single element families and the constraints are equalities.

    Parameters
    ----------
    X : numpy.ndarray
        The data points. (num_datapoints, dim)
    C : numpy.ndarray
        The centroids. (num_clusters, dim)
    cluster_sizes : list
        The size of each cluster. (num_clusters,)
    compressed : bool
        If True, will return a vector of integers instead of a matrix of booleans
    """
    assert C.shape[1] == X.shape[1]
    assert len(cluster_sizes) == C.shape[0]

    num_datapoints = X.shape[0]
    n_clusters = C.shape[0]

    dist = euclidean_distances(X, C, squared=True)

    start_nodes_a = np.arange(num_datapoints)
    end_nodes_a = np.tile(num_datapoints, cluster_sizes[0])
    for i in range(1, n_clusters):
        end_nodes_a = np.concatenate((end_nodes_a, np.tile(num_datapoints + i, cluster_sizes[i])))

    # Create bi-partite graph with costs
    cost_matrix = dist[start_nodes_a[:, None], end_nodes_a - num_datapoints]

    A_bipartite = np.zeros_like(cost_matrix)
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    A_bipartite[row_ind, col_ind] = 1

    cj = end_nodes_a[A_bipartite.argmax(axis=1)] - num_datapoints
    A  = np.zeros((num_datapoints, n_clusters))
    A [np.arange(A_bipartite.shape[0]), cj] = 1

    if compressed:
        A = A.argmax(axis=1)

    return A

def constrained_soft_e_step(X, C, mask, cluster_family_sizes, epsilon=1, n_iter=30):

    n_clusters = C.shape[0]
    cluster_families = sorted(list(cluster_family_sizes.keys()))

    A = [- euclidean_distances(X_, C, squared=True) / epsilon for X_ in X]
    for _ in range(n_iter):

        # Project onto the first simplex with constraints sum_j(wij) = size[i]
        Z = np.full((len(cluster_family_sizes),), - np.inf, dtype=float)

        for A_ in A:
            for i, clusters in enumerate(cluster_families):
                Z[i] = np.logaddexp(Z[i], logsumexp(A_[:, list(clusters)].flatten(), axis=0))

        for A_ in A:
            for i, clusters in enumerate(cluster_families):
                A_[:, list(clusters)] -= Z[i]
                A_[:, list(clusters)] += np.log(cluster_family_sizes[clusters])


        # Project onto the second simplex with constraints sum_i(wij) = 1
        old_A = A
        A = []
        for A_, mask_ in zip(old_A, mask):
            A_[~mask_] = - np.inf

            A_ = log_softmax(A_, axis=1)
            A.append(A_)

    A = [np.exp(A_) for A_ in A]
    return A


def sparsemax_feature_weights_e_step(X, C, A, temp):
    dim = X[0].shape[1]
    num_clusters = C.shape[0]

    a = np.zeros((dim,), dtype=float)
    # calculate the distances to each cluster center
    if A[0].ndim == 1:
        for i in range(num_clusters):
            for X_, A_ in zip(X, A):
                a -= ((X_[A_ == i] - C[i:i+1]) ** 2).sum(axis=0)
    else:
        for X_, A_ in zip(X, A):
            a -= (A_[:, :, np.newaxis] * (X_[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2).sum(axis=(0, 1))

    sorted_a = np.sort(a)[::-1]
    t = (sorted_a[0] - sorted_a[-1]) * len(a)

    return projection_simplex_sort(a / (t))


def sparse_feature_weights_e_step(X, C, A, s, tol, max_iter=20):
    num_datapoints = sum(X_.shape[0] for X_ in X)
    dim = X[0].shape[1]
    num_clusters = C.shape[0]

    # calculating the global dispersion
    # need to do it by hand to not sum over features
    X_all = np.concatenate(X, axis=0)
    a = ((X_all - X_all.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)

    # calculate the distances to each cluster center
    if A[0].ndim == 1:
        for i in range(num_clusters):
            for X_, A_ in zip(X, A):
                a -= ((X_[A_ == i] - C[i:i+1]) ** 2).sum(axis=0)
    else:
        for X_, A_ in zip(X, A):
            a -= (A_[:, :, np.newaxis] * (X_[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2).sum(axis=(0, 1))
    
    # a should be already positive so no need to do clipping
    # assert (a >= 0).all(), "coefs are not all positive"
    a = np.clip(a, a_min=0, a_max=None)

    w = normalized_soft_thresholding(a, 0)

    # the constraint is satisfied
    if np.linalg.norm(w, ord=1) <= s:
        return w


    # otherwise we find the value of the threshold with binary search
    delta_max = a.max()
    delta_min = 0

    for _ in range(max_iter):
        delta_mid = (delta_max + delta_min) / 2
        w = normalized_soft_thresholding(a, delta_mid)

        l1_norm = np.linalg.norm(w, ord=1)
        if l1_norm > s:
            delta_min = delta_mid
        elif l1_norm < s:
            delta_max = delta_mid
        else:
            return w

        if delta_max - delta_min < tol:
            return w

    delta_mid = (delta_max + delta_min) / 2
    w = normalized_soft_thresholding(a, delta_mid)

    return w


def nonsparse_feature_weights_e_step(X, C, A):
    num_datapoints = sum(X_.shape[0] for X_ in X)
    dim = X[0].shape[1]
    num_clusters = C.shape[0]

    # calculating the global dispersion
    # need to do it by hand to not sum over features
    X_all = np.concatenate(X, axis=0)
    a = ((X_all - X_all.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)

    # calculate the distances to each cluster center
    if A[0].ndim == 1:
        for i in range(num_clusters):
            for X_, A_ in zip(X, A):
                a -= ((X_[A_ == i] - C[i:i+1]) ** 2).sum(axis=0)
    else:
        for X_, A_ in zip(X, A):
            a -= (A_[:, :, np.newaxis] * (X_[:, np.newaxis, :] - C[np.newaxis, :, :]) ** 2).sum(axis=(0, 1))

    # a should be already positive so no need to do clipping
    # assert (a >= 0).all(), "coefs are not all positive"
    a = np.clip(a, a_min=0, a_max=None)

    return a / np.linalg.norm(a)
