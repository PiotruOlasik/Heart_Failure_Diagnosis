# Hand-made Naive Bayes
from math import sqrt, pi, exp
import numpy as np
import pandas as pd

class NaiveBayes:

    def gaussian(self, x, mean, std):
        if std == 0:
            return 1.0 if x == mean else 0.0
        exponent = exp(-((x - mean) ** 2.0) / (2.0 * std ** 2.0))
        return (1 / (sqrt(2.0 * pi) * std)) * exponent

    def summarize_by_class(self, X, y):
        summaries = {}
        for cls in np.unique(y):
            X_cls = X[y == cls]
            means = X_cls.mean(axis=0)
            stds = X_cls.std(axis=0)
            summaries[cls] = list(zip(means, stds))
        return summaries

    def calculate_class_probabilities(self, summaries, input_vector, class_priors):
        probabilities = {}
        for cls, feature_summaries in summaries.items():
            prob = class_priors[cls]
            for i in range(len(input_vector)):
                mean, std = feature_summaries[i]
                prob *= self.gaussian(input_vector[i], mean, std)  # P(Xi|Class)
            probabilities[cls] = prob
        return probabilities

    def predict(self, summaries, input_vector, class_priors):
        probs = self.calculate_class_probabilities(summaries, input_vector, class_priors)  # dictionary cls: prob
        return max(probs, key=probs.get)

    def get_predictions(self, X_train, Y_train, X_test):
        summaries = self.summarize_by_class(X_train, Y_train)
        class_priors = {cls: np.mean(Y_train == cls) for cls in np.unique(Y_train)}
        preds = [self.predict(summaries, row, class_priors) for row in X_test]
        return pd.Series(preds)