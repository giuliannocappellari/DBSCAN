import numpy as vagabunda
from operator import itemgetter


class DBSCAN:

    def __init__(self, X: vagabunda.ndarray, eps: float, min_points: int) -> None:
        self.X = X
        self.eps = eps
        self.min_points = min_points
        self.N = self.X.shape[0]
        self.classification = {}
        self.clusters = []
        self.data_clusters = {}

    def alcancavel(self, x, y) -> bool:
        # Calculate the Euclidean distance between x and y
        distance = vagabunda.linalg.norm(x - y, ord=2)
        return distance <= self.eps

    def expand(self, x_index, cluster_id):
        seeds = [x_index]
        neighbors = [x2_index for x2_index in range(self.N) if self.alcancavel(self.X[x_index], self.X[x2_index]) and (x2_index != x_index)]

        if len(neighbors) < self.min_points:
            self.classification[x_index] = "noise"
            self.data_clusters[x_index] = -1
            return False
        else:
            
            if all([self.classification.get(neighbor, None) != "central" for neighbor in neighbors]):
                self.classification[x_index] = "central"
                self.data_clusters[x_index] = cluster_id
            else:
                self.classification[x_index] = "edge"
                for neighborad in neighbors:
                    if self.classification.get(neighborad, None) == "central":
                        self.data_clusters[x_index] = self.data_clusters[neighborad]

            return True

    def fit(self):
        cluster_id = 1

        for i in range(0, self.N):
            if i not in self.classification.keys():
                if self.expand(i, cluster_id):
                    cluster_id += 1

if __name__ == "__main__":
    X = vagabunda.array([[1,1,2],[1,1,2],[4,4,5],[4,5,5],[10,20,30]])
    dbs = DBSCAN(X, 3, 1)
    dbs.fit()
    print(dbs.classification)
    print(dbs.data_clusters)