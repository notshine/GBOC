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

import os
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE


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

def divide_gb_k(data, indices, k):
    
    kmeans = KMeans(n_clusters=k, random_state=5)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    gb_list_temp = []
    for idx in range(k):
        cluster1 = [indices[i] for i in range(len(data)) if labels[i] == idx]
        gb_list_temp.append(np.array(cluster1))

    return gb_list_temp

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
    # plt.legend(fontsize=16)
    plt.savefig("{}.png".format(title))

def compute_ball_info(data, hb_list):

    plt_data = np.concatenate(hb_list)
    
    collected_center = []
    collected_radius = []
    for data_ind in hb_list:
        if len(data) > 1:
            
            center = data[data_ind].mean(0)
            radius = np.max((((data[data_ind] - center) ** 2).sum(axis=1) ** 0.5))

            collected_center.append(center)
            collected_radius.append(radius)
    
    collected_center = np.asarray(collected_center)
    collected_radius = np.asarray(collected_radius)
    
    return collected_center, collected_radius

def spilt_ball_2(data, gb_list):
    n,m = data.shape 
    kmeans = KMeans(n_clusters=2, random_state=0, n_init=3, max_iter=2)
    kmeans.fit(data)
    labels = kmeans.labels_
    
    # 确保返回的是整数数组
    cluster1 = np.array([gb_list[i] for i in range(n) if labels[i] == 0], dtype=np.int64)
    cluster2 = np.array([gb_list[i] for i in range(n) if labels[i] == 1], dtype=np.int64)
    
    # 处理可能的空数组情况
    if cluster1.size == 0 or cluster2.size == 0:
        # 如果其中一个簇为空，则不分割
        return gb_list, gb_list
        
    return cluster1, cluster2


def get_density_volume(data, gb):
    num = len(gb)
    center = data.mean(0)
    diffMat = np.tile(center, (num, 1)) - data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    sum_radius = 0
    radius = max(distances)
    for i in distances:
        sum_radius = sum_radius + i
    mean_radius = sum_radius/num
    dimension = np.shape(data)[1]
    if mean_radius!=0:
        density_volume = num/(sum_radius)
    else:
        density_volume = num

    return density_volume

def division_2_2(data, gb_list, n):

    gb_list_new_2 = []
    for gb in gb_list:
        if(len(gb) >=20):
            # 确保gb是整数类型
            gb = np.asarray(gb, dtype=np.int64)
            
            ball_1, ball_2 = spilt_ball_2(data[gb], gb)
            
            # 确保ball_1和ball_2是整数类型
            if isinstance(ball_1, np.ndarray) and ball_1.size > 0:
                ball_1 = np.asarray(ball_1, dtype=np.int64)
            else:
                # 处理空数组或无效情况
                gb_list_new_2.append(gb)
                continue
                
            if isinstance(ball_2, np.ndarray) and ball_2.size > 0:
                ball_2 = np.asarray(ball_2, dtype=np.int64)
            else:
                # 处理空数组或无效情况
                gb_list_new_2.append(gb)
                continue
                
            density_parent = get_density_volume(data[gb], gb)
            
            # 添加类型和边界检查
            try:
                density_child_1 = get_density_volume(data[ball_1], ball_1)
                density_child_2 = get_density_volume(data[ball_2], ball_2)
            except IndexError:
                # 处理索引错误
                gb_list_new_2.append(gb)
                continue
                
            w = len(ball_1)+len(ball_2)
            w1 = len(ball_1)/w
            w2 = len(ball_2)/w
            w_child = (w1*density_child_1+w2*density_child_2)
            t1 = ((density_child_1 > density_parent)&(density_child_2 > density_parent))
            t2 = (w_child > density_parent)
            t3 = ((len(ball_1) >4) & (len(ball_2) >4))
            if (t2):
                gb_list_new_2.extend([ball_1,ball_2])
            else:
                gb_list_new_2.append(gb)
        else:
          gb_list_new_2.append(gb)  
        
    return gb_list_new_2

def get_radius(data, gb):
    num = len(gb)
    center = data.mean(0)
    diffMat = np.tile(center, (num, 1)) - data
    sqDiffMat = diffMat ** 2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances ** 0.5
    radius = max(distances)
    return radius

def normalized_ball(data, gb_list,radius_detect):
    
    gb_list_temp = []
    for gb in gb_list:
        if (len(gb)<=8):
            pass
            # gb_list_temp.append(gb)
        else:
            ball_1,ball_2 = spilt_ball_2(data[gb], gb)
            if(get_radius(data[gb], gb) <= 2*radius_detect):
                gb_list_temp.append(gb)
            else:
                gb_list_temp.extend([ball_1,ball_2])
    
    return gb_list_temp

def func_granularball_computing(data, visual=False, save_path="", labels=None):
    
    # print(np.shape(data))

    data_num = data.shape[0]
    k1 = int(np.sqrt(len(data)))
    indices = list(range(np.shape(data)[0]))

    gb_list_temp = divide_gb_k(data, indices, k1)
    centers, radius = compute_ball_info(data, gb_list_temp)
    init_centers = centers

    if np.shape(data)[1]==2 and visual:
        plot_balls(data, gb_list_temp, centers, radius, title='Stage0', save_path=save_path, labels=labels)

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = division_2_2(data, gb_list_temp, data_num) #质量轮
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break
    
    centers, radius = compute_ball_info(data, gb_list_temp)

    if np.shape(data)[1]==2 and visual:
        plot_balls(data, gb_list_temp, centers, radius, title='StageBallGeneration', save_path=save_path, labels=labels)

    radius_median = np.median(radius)
    radius_mean = np.mean(radius)
    radius_detect = max(radius_median,radius_mean)

    while 1:
        ball_number_old = len(gb_list_temp)
        gb_list_temp = normalized_ball(data, gb_list_temp, radius_detect)
        ball_number_new = len(gb_list_temp)
        if ball_number_new == ball_number_old:
            break

    centers, radius = compute_ball_info(data, gb_list_temp)

    if np.shape(data)[1]==2 and visual:
        plot_balls(data, gb_list_temp, centers, radius, title='StageBallClean', save_path=save_path, labels=labels)

    # print("centers", np.shape(centers))
    return centers, gb_list_temp, init_centers

def update_legend_marker_size(handle, orig):
    "Customize size of the legend marker"
    handle.update_from(orig)
    handle.set_sizes([20])


def tsne_visualization(data, centers, gb_list, save_path="", save_name=""):

    # data: N, C
    # centers: K, C

    N = np.shape(data)[0]
    if N > 1000:
        train_start_point_set = list(range(0, N))
        batch_indices = random.sample(train_start_point_set, 1000) 
        data = data[batch_indices]
        N = 1000
    all_data = np.concatenate([data, centers], axis=0)
    X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(all_data)
    
    plt.figure(figsize=(10, 9))
    plt.scatter(X_embedded[:N, 0], X_embedded[:N, 1], s=30, c="black", linewidths=1, alpha=1, marker='o', label='data point')
    plt.scatter(X_embedded[N:, 0], X_embedded[N:, 1], s=30, c="red", linewidths=1, alpha=1, marker='o', label='granular-ball centers')
    plt.savefig(os.path.join(save_path, "{}.png".format(save_name)))



if __name__ == '__main__':
    
    # np.random.seed(42)
    # X_inliers = 0.3 * np.random.randn(100, 2)
    # X_inliers = np.r_[X_inliers + 2, X_inliers - 2]
    # X_outliers = np.random.uniform(low=-4, high=4, size=(20, 2))
    # X = np.r_[X_inliers, X_outliers]
    # n_outliers = len(X_outliers)

    n_samples = 1000
    n_outliers = 20
    X = 4.0*(make_moons(n_samples=n_samples, noise=0.05, random_state=0)[0] - np.array([0.5, 0.25]))
    rng = np.random.RandomState(42)
    X = np.concatenate([X, rng.uniform(low=-6, high=6, size=(n_outliers, 2))], axis=0)

    ground_truth = np.ones(len(X), dtype=int)
    ground_truth[-n_outliers:] = -1

    centers, gb_list_temp, init_centers = func_granularball_computing(X, visual=True, labels=ground_truth)

    print(np.shape(init_centers), np.shape(centers))

    # """
    # clf = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
    # y_pred = clf.fit_predict(X)
    # n_errors = (y_pred != ground_truth).sum()
    # X_scores = clf.negative_outlier_factor_

    # plt.figure(figsize=(12,6))
    # plt.scatter(X[:, 0], X[:, 1], color="k", s=3.0, label="Data points")
    # radius = (X_scores.max() - X_scores) / (X_scores.max() - X_scores.min())
    # scatter = plt.scatter(
    #     X[:, 0],
    #     X[:, 1],
    #     s=1000 * radius,
    #     edgecolors="r",
    #     facecolors="none",
    #     label="Outlier scores",
    # )
    # plt.axis("tight")
    # # plt.xlim((-5, 5))
    # # plt.ylim((-5, 5))
    # plt.xlabel("prediction errors: %d" % (n_errors))
    # plt.legend(
    #     handler_map={scatter: HandlerPathCollection(update_func=update_legend_marker_size)}
    # )
    # plt.title("Local Outlier Factor (LOF)")
    # plt.show()


