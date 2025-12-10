

import sklearn
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import LocalOutlierFactor
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import numpy as np



def func_granularball_computing():
    pass



def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])

if __name__ == '__main__':
    
    np.random.seed(42)

    X_inliers = 0.3 * np.random.randn(100, 2)
    X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    X = np.r_[X_inliers, X_outliers]

    n_outliers = len(X_outliers)
    ground_truth = np.ones(len(X), dtype=int)
    ground_truth[-n_outliers:] = -1

    clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    y_pred = clf.fit_predict(X)
    n_errors = (y_pred != ground_truth).sum()
    X_scores = clf.negative_outlier_factor_

    plt.figure(figsize=(12,6))
    plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
    radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        s=1000 * radius,
        edgecolors="r",
        facecolors="none",
        label="Outlier scores",
    )
    plt.axis("tight")
    plt.xlim((-5, 5))
    plt.ylim((-5, 5))
    plt.xlabel("prediction errors: %d" % (n_errors))
    plt.legend(
        handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
    )
    plt.title("Local Outlier Factor (LOF)")
    plt.show()

