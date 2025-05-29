# Hand-made KNN:
import math
import pandas as pd

class KNNClass:
    def metric(self, a, b):
        sum = 0
        if len(a) == len(b):

            # Canberra metric
            for i in range(len(a)):
                denom = math.fabs(a[i]) + math.fabs(b[i])
                if denom != 0:
                    sum += math.fabs(a[i] - b[i]) / denom
                else:
                    sum += 1

            return sum
        else:
            print("Different dimensions")

    def knn(self, X_train, Y_train, X_test, k):
        Y_prediction = []
        Y_train_np = Y_train.to_numpy()

        for X_test_vector in X_test:
            distances = []

            for i in range(len(X_train)):
                dist = self.metric(X_test_vector, X_train[i])
                distances.append((dist, Y_train_np[i]))

            distances.sort(key=lambda x: x[0])
            k_neighbors = distances[:k]  # list of tuples

            labels = [neighbor[1] for neighbor in k_neighbors]

            control = set(labels)
            most_common = ["", 0]
            for item in control:
                if (labels.count(item) > most_common[1]):
                    most_common = [item, labels.count(item)]

            Y_prediction.append(most_common[0])

        return pd.Series(Y_prediction)