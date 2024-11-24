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
            cp.Minimize(cp.sum(slack)),
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
    def __init__(self, ratio: float):
        self.ratio = ratio

    def select(self, train_x: np.ndarray):
        return data_to_label
