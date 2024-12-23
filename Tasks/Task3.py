import numpy as np
import cvxpy as cp


class MyLabelSelection:
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
        n_samples = trainX.shape[0]
        self.n_samples_to_label = int(np.ceil(self.ratio * n_samples))

        from sklearn.metrics.pairwise import euclidean_distances

        # distance matrix D is an n x n matrix where D_{ij} = dist(x_i, x_j)
        distance_matrix = euclidean_distances(trainX, trainX)

        if self.epsilon is None:
            # should be on ratio?
            self.epsilon = np.percentile(distance_matrix, 5)
            print(f"Epsilon: {self.epsilon}")

        # for each x_j, its neighborhood is N_j = {i | D_{ij} <= epsilon}
        neighborhoods = (distance_matrix <= self.epsilon).astype(int)

        coverage_counts = np.sum(neighborhoods, axis=1)
        if self.ratio == 0.05:
            low_density_filter = (coverage_counts > 8).astype(int)
        if self.ratio == 0.1:
            low_density_filter = (coverage_counts > 4).astype(int)
        if self.ratio == 0.2:
            low_density_filter = (coverage_counts > 5).astype(int)
        if self.ratio == 0.5:
            low_density_filter = (coverage_counts > -1).astype(int)
        if self.ratio == 1.0:
            low_density_filter = (coverage_counts > -1).astype(int)
        print(f"Low-density filter applied. {np.sum(low_density_filter == 0)} points excluded.")

        # coverage counts
        w = np.sum(neighborhoods, axis=1)
        self.w = w

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
        #objective = cp.Maximize(cp.sum(y))
        #objective = cp.Maximize(cp.sum(cp.multiply(density_scores, y)))

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
