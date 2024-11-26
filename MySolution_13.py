import numpy as np
import cvxpy as cp
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
from scipy.spatial.distance import cdist


class MyClassifier:
    def __init__(self, class_num: int):
        self.class_num = class_num  # number of classes
        self.w = None
        self.b = None
        self.cc_inverse = None  # Used for decompressing the coordinate compression

    def train(self, train_x: np.ndarray, train_y: np.ndarray):
        n, m = train_x.shape
        # Coordinate compression
        self.cc_inverse, train_y_cc = np.unique(train_y, return_inverse=True)

        y_1hot = np.zeros((n, self.class_num))
        y_1hot[np.arange(n), train_y_cc.astype(int)] = 1

        self.w = cp.Variable((m, self.class_num))
        self.b = cp.Variable((self.class_num, 1))
        slack = cp.Variable((n, self.class_num))

        prob = cp.Problem(
            cp.Minimize(cp.sum(slack) / n),
            [
                slack >= 0,
                train_x @ self.w + cp.sum(self.b.T) - y_1hot <= slack,
                train_x @ self.w + cp.sum(self.b.T) - y_1hot >= -slack,
            ],
        )
        prob.solve()

    def predict(self, test_x: np.ndarray) -> np.ndarray:
        n, m = test_x.shape
        # Return the predicted class labels of the input data (testX)
        pred_y = test_x @ self.w.value + np.ones((n, 1)) @ self.b.value.T
        pred_y = np.argmax(pred_y, axis=1)
        return self.cc_inverse[pred_y]

    def evaluate(self, test_x: np.ndarray, test_y: np.ndarray) -> float:
        pred_y = self.predict(test_x)
        return accuracy_score(test_y, pred_y)


class MyClustering:
    def __init__(self, class_num: int, iters: int = 100):
        self.class_num = class_num
        self.labels = None
        self.centers = None
        self.iters = iters

    def train(self, train_x: np.ndarray):
        n, _ = train_x.shape
        self.centers = train_x[np.random.choice(n, self.class_num, replace=False), :]
        self.labels = np.full(n, -1, dtype=int)

        for i in range(self.iters):
            d = cdist(train_x, self.centers, metric="euclidean")
            y = cp.Variable(self.class_num)
            x = cp.Variable((n, self.class_num))

            obj = cp.Minimize(cp.sum(cp.multiply(d, x)))
            constraints = [
                cp.sum(x, axis=1) == 1,
                x <= cp.reshape(y, (1, self.class_num)),
                cp.sum(y) == self.class_num,
                x >= 0,
                x <= 1,
                y >= 0,
                y <= 1,
            ]
            prob = cp.Problem(obj, constraints)
            prob.solve()

            new_labels = np.argmax(x.value, axis=1)

            if np.array_equal(new_labels, self.labels):
                break

            self.labels = new_labels
            self.centers = np.array(
                [
                    np.median(train_x[self.labels == k], axis=0)
                    for k in range(self.class_num)
                ]
            )

        return self.labels

    def infer_cluster(self, test_x: np.ndarray):
        dists = cdist(test_x, self.centers, metric="euclidean")
        return np.argmin(dists, axis=1)

    def evaluate_clustering(self, train_y: np.ndarray):
        label_reference = self.get_class_cluster_reference(self.labels, train_y)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        return normalized_mutual_info_score(train_y, aligned_labels)

    def evaluate_classification(
        self, train_y: np.ndarray, test_x: np.ndarray, test_y: np.ndarray
    ) -> float:
        pred_labels = self.infer_cluster(test_x)
        label_reference = self.get_class_cluster_reference(self.labels, train_y)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        return accuracy_score(test_y, aligned_labels)

    def get_class_cluster_reference(
        self, cluster_labels: np.ndarray, true_labels: np.ndarray
    ):
        """Assign a class label to each cluster using majority vote"""
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i, 1, 0)
            num = np.bincount(true_labels[index == 1]).argmax()
            label_reference[i] = num
        return label_reference

    def align_cluster_labels(self, cluster_labels: np.ndarray, reference):
        """update the cluster labels to match the class labels"""
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]
        return aligned_lables


class MyLabelSelection:
    """Task 3 (Option 1)"""

    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        self.epsilon = None
        self.n_samples_to_label = None
        self.w = None

    def select(self, trainX):
        """
        The LP:
        selection variables z_i in {0, 1} for i = 1,...,n
        z_i = 1 if we select x_i to label, 0 otherwise

        y_j in {0, 1} for j = 1,...,n
        y_j = 1 if sample x_j is "covered" by a selected sample (distance eps of at least one labeled sample)

        D is an n x n matrix where D_ij = dist(x_i, x_j)
        for each sample x_j, its neighborhood N_j = {i | D_ij <= eps}

        then,
        max sum(y)
        s t sum(z) = L
        y_j <= sum_{i in N_j} z_i forall j (y can only be 1 if at least one z_i is its neighborhood; one labeled sample covers x_j)
        z_i in [0, 1]
        y_j in [0, 1]
        """
        """Task 3-2"""

        n_samples = trainX.shape[0]
        M = trainX.shape[1]
        self.n_samples_to_label = int(np.ceil(self.ratio * n_samples))

        from sklearn.metrics.pairwise import euclidean_distances

        # distance matrix D is an n x n matrix where D_{ij} = dist(x_i, x_j)
        distance_matrix = euclidean_distances(trainX, trainX)

        if self.epsilon is None:
            # should be on ratio?
            if M == 2:
                self.epsilon = np.percentile(distance_matrix, 5)
            if M == 784:
                self.epsilon = np.percentile(distance_matrix, 10)
            print(f"Epsilon: {self.epsilon}")

        # for each x_j, its neighborhood is N_j = {i | D_{ij} <= epsilon}
        neighborhoods = (distance_matrix <= self.epsilon).astype(int)

        coverage_counts = np.sum(neighborhoods, axis=1)
        if M == 2:
            if self.ratio == 0.05:
                low_density_filter = (coverage_counts > 100).astype(int)
            if self.ratio == 0.1:
                low_density_filter = (coverage_counts > 90).astype(int)
            if self.ratio == 0.2:
                low_density_filter = (coverage_counts > 80).astype(int)
            if self.ratio == 0.5:
                low_density_filter = (coverage_counts > 70).astype(int)
            if self.ratio == 1.0:
                low_density_filter = (coverage_counts > -1).astype(int)
            print(f"Low-density filter applied. {np.sum(low_density_filter == 0)} points excluded.")
        if M == 784:
            # Don't use coverage filtering
            if self.ratio == 0.05:
                low_density_filter = (coverage_counts > -1).astype(int)
            if self.ratio == 0.1:
                low_density_filter = (coverage_counts > -1).astype(int)
            if self.ratio == 0.2:
                low_density_filter = (coverage_counts > -1).astype(int)
            if self.ratio == 0.5:
                low_density_filter = (coverage_counts > -1).astype(int)
            if self.ratio == 1.0:
                low_density_filter = (coverage_counts > -1).astype(int)
            print(f"Low-density filter applied. {np.sum(low_density_filter == 0)} points excluded.")

        # coverage counts
        w = np.sum(neighborhoods, axis=1)
        self.w = w

        # potential alternative
        # density_scores = np.sum(neighborhoods, axis=1)
        # density_scores = np.log(density_scores + 1)
        # density_scores = density_scores / np.max(density_scores)

        z = cp.Variable(n_samples)
        y = cp.Variable(n_samples)

        constraints = []

        constraints.append(cp.sum(z) == self.n_samples_to_label)

        # z is our selecting labels
        # 1 if we select the same x_i to label
        constraints.append(z >= 0)
        constraints.append(z <= 1)

        # y is our coverage constraints
        # y_j = 1 if sample x_j is covered by the selected sample
        # i.e., distance epsilon of at least one labeled sample
        constraints.append(y >= 0)
        constraints.append(y <= 1)

        # this is our coverage constraint: y_j \leq \sum_{i \in N_j} z_i \forall j = 1,...,n
        for j in range(n_samples):
            # y_j <= sum_{i in N_j} z_i
            indices_in_neighborhood = np.where(neighborhoods[j])[0]
            if len(indices_in_neighborhood) > 0:
                constraints.append(y[j] <= cp.sum(z[indices_in_neighborhood]))
            else:
                constraints.append(y[j] == 0)  # no neighbors in epsilon

        z = cp.multiply(low_density_filter, z)

        objective = cp.Maximize(cp.sum(y))

        # Potential alternative
        # objective = cp.Maximize(cp.sum(cp.multiply(density_scores, y)))

        prob = cp.Problem(objective, constraints)
        prob.solve()
        print(f"Optimal value from label selection:{prob.value}")

        z_lp = z.value
        z_lp = np.clip(z_lp, 0, 1)

        z_rounded = np.random.binomial(1, z_lp)

        current_labels = np.sum(z_rounded)
        if current_labels > self.n_samples_to_label:
            indices = np.where(z_rounded == 1)[0]
            excess = int(current_labels - self.n_samples_to_label)
            to_zero = np.random.choice(indices, size=excess, replace=False)
            z_rounded[to_zero] = 0
        elif current_labels < self.n_samples_to_label:
            indices = np.where(z_rounded == 0)[0]
            deficit = int(self.n_samples_to_label - current_labels)
            to_one = np.random.choice(indices, size=deficit, replace=False)
            z_rounded[to_one] = 1

        data_to_label = np.where(z_rounded == 1)[0]

        return data_to_label

class MyLabelSelection_Clustering:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label

    def select(self, train_x: np.ndarray):
        take_num = int(len(train_x) * self.ratio)
        cluster = MyClustering(take_num, 10)
        labels = cluster.train(train_x)
        centers = cluster.centers

        inds = np.full(take_num, -1)
        for t in range(take_num):
            group = np.where(labels == t)[0]
            dists = np.linalg.norm(train_x[group] - centers[t], axis=1)
            inds[t] = group[np.argmin(dists)]

        return inds
