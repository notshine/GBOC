

import sklearn
from sklearn.datasets import make_blobs, make_moons
from sklearn.neighbors import LocalOutlierFactor
import time

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerPathCollection
import numpy as np
import torch
import random

from sklearn.cluster import KMeans

def calculate_distances(data, p):
    if isinstance(data, torch.Tensor) and isinstance(p, torch.Tensor):
        dis = (data - p).clone().detach() ** 2
        dis = dis.cpu().numpy()
    else:
        dis = (data - p) ** 2
    dis_top10 = np.sort(dis)[-10:]

    return 0.6 * np.sqrt(dis).sum() + 0.4 * np.sqrt(dis_top10).sum()

def splits(args, gb_dict, select):

    pass

def divide_gb_k(data, indices, k):
    
    kmeans = KMeans(n_clusters=k, random_state=5)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    gb_list_temp = []
    for idx in range(k):
        cluster1 = [indices[i] for i in range(len(data)) if labels[i] == idx]
        gb_list_temp.append(np.array(cluster1))

    return gb_list_temp

def func_granularball_computing(data, metric="max", visual=False, save_path="", labels=None):
    
    # print(np.shape(data))
    k1 = int(np.sqrt(len(data)))
    indices = list(range(np.shape(data)[0]))

    gb_list_temp = divide_gb_k(data, indices, k1) # First, roughly divide it into √n spheres.

    centers, radius = compute_ball_info(data, gb_list_temp, metric=metric)

    if np.shape(data)[1]==2 and visual:
        plot_balls(data, gb_list_temp, centers, radius, title='Stage 0', save_path=save_path, labels=labels)

    gb_list_not_temp = []
    while 1:

        ball_number_old = len(gb_list_temp) + len(gb_list_not_temp)
        gb_list_temp, gb_list_not_temp = division_central_consistency(data, gb_list_temp, gb_list_not_temp, metric=metric)

        ball_number_new = len(gb_list_temp) + len(gb_list_not_temp)
        
        if ball_number_new == ball_number_old:
            gb_list_temp = gb_list_not_temp
            break

    centers, radius = compute_ball_info(data, gb_list_temp, metric=metric)
    if np.shape(data)[1]==2 and visual:
        plot_balls(data, gb_list_temp, centers, radius, title='Stage End', save_path=save_path, labels=labels)



def spilt_ball_k_means(data, gb_list, n):

    kmeans = KMeans(n_clusters=n, random_state=0, n_init=3, max_iter=2)
    kmeans.fit(data)
    labels = kmeans.labels_
    cluster1 = [gb_list[i].tolist() for i in range(len(data)) if labels[i] == 0]
    cluster2 = [gb_list[i].tolist() for i in range(len(data)) if labels[i] == 1]
    ball1 = np.array(cluster1)
    ball2 = np.array(cluster2)

    return [ball1, ball2]
  
def get_dm_sparse(data, list_gb, metric="max"):
    num = len(data)
    dim = np.shape(data)[1]

    center = data.mean(0)
    diff_mat = center - data
    sq_diff_mat = diff_mat ** 2
    sq_distances = sq_diff_mat.sum(axis=1)
    distances = sq_distances ** 0.5

    if metric=="max":
        radius = np.max(distances)
    if metric=="mean":
        radius = np.mean(distances)
    if metric=="median":
        radius = np.median(distances)
    sparsity = num / (radius ** dim)
    
    return sparsity

def division_central_consistency(data, gb_list, gb_list_not, metric="max"):
    ''' Central consistency division
    Args:
        gb_list:
        gb_list_not:
    Returns:
    '''
    gb_list_new = []

    for gb in gb_list:
        if len(gb) > 1:
            list_ball_1, list_ball_2 = spilt_ball_k_means(data[gb], gb, 2)
            
            # print(list_ball_1, list_ball_2)
            sprase_parent = get_dm_sparse(data[gb], gb, metric=metric)
            sprase_child1 = get_dm_sparse(data[list_ball_1], list_ball_1, metric=metric)
            sprase_child2 = get_dm_sparse(data[list_ball_2], list_ball_2, metric=metric)
            print("sprase_parent", sprase_parent, sprase_child1, sprase_child2)

            flag_split = False
            # if len(list_ball_1) == 1 and len(list_ball_2) > 1:
            #     if sprase_child2 >= sprase_parent: 
            #         flag_split = True
            # elif len(list_ball_2) == 1 and len(list_ball_1) > 1:
            #     if sprase_child1 >= sprase_parent: 
            #         flag_split = True
            if len(list_ball_1) > 1 and len(list_ball_2) > 1:
                # if (sprase_child1 >= sprase_parent or sprase_child2 >= sprase_parent):
                if sprase_child1 + sprase_child2 >= sprase_parent:
                    flag_split = True

            if flag_split:
                gb_list_new.extend([list_ball_1, list_ball_2])
            else:
                gb_list_not.append(gb)
        else:
            gb_list_not.append(gb)

    return gb_list_new, gb_list_not

def plot_balls(data, gb_list_temp, centers, radius, title = '', save_path="", labels=None):

    plt.figure(figsize=(10, 9))
    if labels is not None:
        NUM_CLASSES = np.unique(labels)
        plt.scatter(data[labels==NUM_CLASSES[0], 0], data[labels==NUM_CLASSES[0], 1], s=7, c="red", linewidths=2, alpha=0.6, marker='o', label='data point')
        plt.scatter(data[labels==NUM_CLASSES[1], 0], data[labels==NUM_CLASSES[1], 1], s=7, c="black", linewidths=2, alpha=0.6, marker='o', label='data point')

    else:
        plt.scatter(data[:, 0], data[:, 1], s=7, c="blue", linewidths=2, alpha=0.6, marker='o', label='data point')
    
    theta = np.arange(0, 2 * np.pi, 0.01)
    for i, indices in enumerate(gb_list_temp):
        x = centers[i][0] + radius[i] * np.cos(theta)
        y = centers[i][1] + radius[i] * np.sin(theta)

        plt.plot(x, y, ls='-', color='black', lw=0.7)

    # plt.title(title)
    plt.legend(fontsize=16)
    plt.savefig("{}.png".format(title))

def compute_ball_info(data, hb_list, metric="max"):

    plt_data = np.concatenate(hb_list)
    
    collected_center = []
    collected_radius = []
    for data_ind in hb_list:
        if len(data) > 1:
            
            center = data[data_ind].mean(0)
            if metric == "max":
                radius = np.max((((data[data_ind] - center) ** 2).sum(axis=1) ** 0.5))
            elif metric == "mean":
                radius = np.mean((((data[data_ind] - center) ** 2).sum(axis=1) ** 0.5))
            elif metric == "median":
                radius = np.median((((data[data_ind] - center) ** 2).sum(axis=1) ** 0.5))

            collected_center.append(center)
            collected_radius.append(radius)
    
    return collected_center, collected_radius

def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])


if __name__ == '__main__':
    
    # np.random.seed(42)
    # X_inliers = 0.3 * np.random.randn(100, 2)
    # X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    # X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    # X = np.r_[X_inliers, X_outliers]
    # n_outliers = len(X_outliers)

    n_samples = 300
    n_outliers = 40
    X = 4.0*(make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0] - np.array([0.5, 0.25]))
    rng = np.random.RandomState(42)
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

    ground_truth = np.ones(len(X), dtype=int)
    ground_truth[-n_outliers:] = -1

    func_granularball_computing(X, metric="mean", visual=True, labels=ground_truth)

    
    """
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
    # plt.xlim((-5, 5))
    # plt.ylim((-5, 5))
    plt.xlabel("prediction errors: %d" % (n_errors))
    plt.legend(
        handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
    )
    plt.title("Local Outlier Factor (LOF)")
    plt.show()
    """


