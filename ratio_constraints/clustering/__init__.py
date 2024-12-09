import sys
import time

import numpy as np

from sklearn.metrics.pairwise import euclidean_distances

from ratio_constraints.clustering import e_step


from ratio_constraints.clustering.utils import scale, m_step, initialize_centroids, test_cluster_family_sizes, fda_step


class LDAKMeans:
    def __init__(self, assignments_type, n_clusters, init='k-means++', random_state=None, device="cpu"):
        assert assignments_type in ["soft", "hard"]

        self.assignments_type = assignments_type

        self.n_clusters = n_clusters
        self.random_state = random_state
        self.init = init
        self.device = device

    def calculate_labels(self, A):
        if A.ndim == 1:
            return A
        else:
            return A.argmax(axis=1)

    def fit(
        self,
        X,
        y,
        tag2clusters,
        cluster2tag,
        true_y=None,
        evaluate=None,
        cluster_family_sizes=None,
        epsilon=1,
        n_iter=10,
        kmeans_n_iter=10,
        n_iter_bregman=10,
    ):
        print("================================================================", file=sys.stderr, flush=True)
        print("                        {} kmeans".format(self.assignments_type), file=sys.stderr, flush=True)

        assert all(X_.shape[1] == X[0].shape[1] for X_ in X), "Not all sentences have the same number of features per word"
        dim = X[0].shape[1]
        n_total = sum(X_.shape[0] for X_ in X)

        mask = [
            np.vstack(
                [
                    np.isin(np.arange(self.n_clusters), tag2clusters[y__]) if y__ >= 0
                    else np.full((self.n_clusters,), True)
                    for y__ in y_
                ]
            )
            for y_ in y
        ]

        print("=== using LDA", file=sys.stderr, flush=True)

        # in case it is not constrained
        if cluster_family_sizes is None:
            print("=== non constrained", file=sys.stderr, flush=True)

            if self.assignments_type == "soft":
                _e_step = lambda X, C, mask, **kwargs: e_step.unconstrained_soft_e_step(X, C, mask)
            else:
                _e_step = lambda X, C, mask, **kwargs: e_step.unconstrained_hard_e_step(X, C, mask)

        else:
            print("=== constrained:\n\t{}".format(cluster_family_sizes), file=sys.stderr, flush=True)

            cluster_family_sizes = test_cluster_family_sizes(cluster_family_sizes, self.n_clusters, n_total)

            if self.assignments_type == "soft":
                _e_step = lambda X, C, mask, cluster_family_sizes, epsilon, n_iter: e_step.constrained_soft_e_step(X, C, mask, cluster_family_sizes=cluster_family_sizes, epsilon=epsilon, n_iter=n_iter)
            else:
                _e_step = lambda X, C, mask, cluster_family_sizes, **kwargs: e_step.constrained_hard_e_step(X, C, mask, cluster_family_sizes=cluster_family_sizes)



        # sparse kmeans needs features to be centerd and have 0 mean
        X_all = np.concatenate(X, axis=0)
        y_all = np.concatenate(y, axis=0)


        print("=== centroids initialization... ", end="", file=sys.stderr, flush=True)
        start_time = time.time()
        C = initialize_centroids(X_all, y_all, cluster2tag, tag2clusters, self.init, self.n_clusters)
        print("{} (s)".format(time.time() - start_time), file=sys.stderr, flush=True)

        print(file=sys.stderr, flush=True)

        A = np.empty(0)
        U = np.eye(dim)

        self.n_iter = 0
        n_components = min(self.n_clusters - 1, dim)

        # first training loop
        for i_outer in range(n_iter):
            print("=== iteration n {}".format(self.n_iter), file=sys.stderr, flush=True)
            self.n_iter += 1

            for i_inner in range(kmeans_n_iter):
                print("=== e_step on A... ", end="", file=sys.stderr, flush=True)
                start_time = time.time()
                A = _e_step(
                    [X_ @ U.T for X_ in X],
                    C @ U.T,
                    mask,
                    cluster_family_sizes=cluster_family_sizes,
                    epsilon=epsilon,
                    n_iter=n_iter_bregman
                )
                print("{} (s)".format(time.time() - start_time), file=sys.stderr, flush=True)

                print("=== m_step on C... ", end="", file=sys.stderr, flush=True)
                start_time = time.time()
                C = m_step(X, A, self.n_clusters)
                print("{} (s)".format(time.time() - start_time), file=sys.stderr, flush=True)


            if n_iter > 1:
                print("=== lda step... ", end="", file=sys.stderr, flush=True)
                start_time = time.time()

                A_2d_all = np.concatenate(A if A[0].ndim == 2 else [np.eye(self.n_clusters)[A_] for A_ in A], axis=0)

                U = fda_step(X_all, A_2d_all, C, n_components)

                print("{} (s)".format(time.time() - start_time), file=sys.stderr, flush=True)
            else:
                print("=== no lda step", file=sys.stderr, flush=True)


            obj_value = self.calculate_objective(X, A, C, U)
            print("objective: {}".format(obj_value), file=sys.stderr, flush=True)

            if true_y is not None and evaluate is not None:
                train_score, train_accuracy = self.__evaluate(C, U, evaluate, X, true_y)
                print("train f1-score: {}".format(train_score), file=sys.stderr, flush=True)
                print("train accuracy: {}\n".format(train_accuracy), file=sys.stderr, flush=True)


        self.C = C
        self.A = A
        self.U = U

        print("================================================================\n", file=sys.stderr, flush=True)

        return self

    def __evaluate(self, C, U, evaluate, X, y):
        self.C = C
        self.U = U
        return evaluate(self, X, y)

    def predict_emissions(
        self,
        X
    ):
        assert self.C is not None

        WC = self.C @ self.U.T

        output = []
        for X_ in X:
            output.append(- euclidean_distances(X_ @ self.U.T, WC, squared=True))

        return output

    def calculate_objective(self, X, A, C, U):

        WX = [X_ @ U.T for X_ in X]
        WC = C @ U.T

        output = 0
        if A[0].ndim == 1:
            for i in range(self.n_clusters):
                for A_, WX_ in  zip(A, WX):
                    if (A_ == i).any():
                        output += euclidean_distances(WX_[A_ == i], WC[i: i+1], squared=True).sum()
        else:
            for A_, WX_ in  zip(A, WX):
                output = (A_ * euclidean_distances(WX_, WC, squared=True)).sum()

        return output

