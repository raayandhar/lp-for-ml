import numpy as np
import cvxpy as cp
from sklearn.metrics import normalized_mutual_info_score, accuracy_score


class MyClassifier:
    """Task 1"""

    ### Formulation:
    # minimize || X_train @ W + ones((N, 1)) @ b.T - y_1hot ||
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        self.W = None
        self.b = None
        self.cc_inverse = None # Used for decompressing the coordinate compression

    def train(self, trainX, trainY):
        """
        Task 1-2
        TODO: train classifier using LP(s) and updated parameters needed in your algorithm
        """
        N, M = trainX.shape
        # self.W = np.zeros((M, self.K))
        # self.b = np.zeros((self.K, 1))
        self.cc_inverse, trainY_cc = np.unique(trainY, return_inverse=True) # Coordinate compression

        y_1hot = np.zeros((N, self.K))
        y_1hot[np.arange(N), trainY_cc.astype(int)] = 1

        self.W = cp.Variable((M, self.K))
        self.b = cp.Variable((self.K, 1))
        slack = cp.Variable(1)
        
        prob = cp.Problem(cp.Minimize(slack),
                          [slack >= 0,
                           trainX @ self.W + np.ones((N, 1)) @ self.b.T - y_1hot <= slack,
                           trainX @ self.W + np.ones((N, 1)) @ self.b.T - y_1hot >= -slack
                           ])
        prob.solve()
        print(f"Optimal value:{prob.value}")

    def predict(self, testX):
        """
        Task 1-2
        TODO: predict the class labels of input data (testX) using the trained classifier
        """
        N, M = testX.shape
        # Return the predicted class labels of the input data (testX)
        predY = testX @ self.W.value + np.ones((N, 1)) @ self.b.value.T
        print("predY.shape", predY.shape)
        predY = np.argmax(predY, axis=1)
        return self.cc_inverse[predY]

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy


class MyClustering:
    """Task 2"""

    def __init__(self, K):
        self.K = K  # number of classes
        self.labels = None

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        # self.cluster_centers_ = None

    def train(self, trainX):
        """
        Task 2-2
        TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        """

        # Update and teturn the cluster labels of the training data (trainX)
        return self.labels

    def infer_cluster(self, testX):
        """
        Task 2-2
        TODO: assign new data points to the existing clusters
        """

        # Return the cluster labels of the input data (testX)
        return pred_labels

    def evaluate_clustering(self, trainY):
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
        label_reference = self.get_class_cluster_reference(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy

    def get_class_cluster_reference(self, cluster_labels, true_labels):
        """assign a class label to each cluster using majority vote"""
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i, 1, 0)
            num = np.bincount(true_labels[index == 1]).argmax()
            label_reference[i] = num

        return label_reference

    def align_cluster_labels(self, cluster_labels, reference):
        """update the cluster labels to match the class labels"""
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables


class MyLabelSelection:
    """Task 3 (Option 1)"""

    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        """Task 3-2"""

        # Return an index list that specifies which data points to label
        return data_to_label


class MyFeatureSelection:
    """Task 3 (Option 2)"""

    def __init__(self, num_features):
        self.num_features = num_features  # target number of features
        ### TODO: Initialize other parameters needed in your algorithm

    def construct_new_features(
        self, trainX, trainY=None
    ):  # NOTE: trainY can only be used for construting features for classification task
        """Task 3-2"""

        # Return an index list that specifies which features to keep
        return feat_to_keep