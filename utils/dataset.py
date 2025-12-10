import torch
import torch.utils.data
import numpy as np

from granularball_computing.granularball_generation import *

epsilon = 1e-8


def assign_granular_ball_labels(features, centers):
    """
    根据特征矩阵和粒球中心矩阵，为每个特征点分配最近的粒球中心标签（矩阵形式实现）

    参数:
    features: numpy数组，形状为 (N, D)，表示 N 个样本，每个样本有 D 维特征
    centers: numpy数组，形状为 (C, D)，表示 C 个粒球中心，每个中心有 D 维特征

    返回:
    labels: numpy数组，形状为 (N,)，表示每个样本所属的粒球中心的索引
    """
    # 欧氏距离的平方可以通过以下公式向量化计算：
    # ||x - c||^2 = ||x||^2 + ||c||^2 - 2 * x · c^T

    # print(np.shape(features), np.shape(centers))

    # 计算 ||x||^2，形状 (N, 1)
    features_squared = np.sum(features**2, axis=1, keepdims=True)

    # 计算 ||c||^2，形状 (1, C)
    centers_squared = np.sum(centers**2, axis=1, keepdims=True).T

    # 计算 x · c^T，形状 (N, C)
    cross_term = np.dot(features, centers.T)

    # 欧氏距离的平方矩阵，形状 (N, C)
    squared_distances = features_squared + centers_squared - 2 * cross_term

    # 取平方根得到欧氏距离（可选，如果只需要比较大小可以不取）
    distances = np.sqrt(squared_distances)

    # 对于每个样本，找到距离最小的中心作为其标签
    labels = np.argmin(distances, axis=1)

    return labels

class GB_ReconstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, model, window_size, dis_mode="ball", gb_centers=None, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(0, (self.data.shape[0] - window_size) // stride + 1)
        self.samples, self.targets = self._generate_samples()

        # ===============================================
        self.model = model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.all_features = self.extract_features(self.samples.to(device))

        if dis_mode == "ball":
            if gb_centers is None:    
                self.gb_centers, self.gb_labels = self.obtain_granularball_labels(self.all_features.detach().cpu().numpy())
            else:
                self.gb_centers = gb_centers
                self.gb_labels = self.obtain_granularball_labels_w_centers(self.all_features.detach().cpu().numpy(), self.gb_centers)

        elif dis_mode == "cluster": # fixed K
            pass
        elif dis_mode == "instance":
            pass

        
    def extract_features(self, samples):
        
        # torch.Size([391, 10, 1])
        samples = samples.permute(1, 0, 2)

        # print(np.shape(samples))
        batch_size = 32
        all_features = []
        for i in range(np.shape(samples)[1]):
            ind = i*batch_size
            if ind>np.shape(samples)[1]:
                break
            else:
                batch_inp = samples[:,ind:ind+batch_size]
                if np.shape(batch_inp)[1] == 0:
                    break
                feas = self.model.obtain_features(batch_inp)
                # print("111", ind, np.shape(feas))
                all_features.append(feas)
        all_features = torch.cat(all_features, dim=0)
        # print("all_features", np.shape(all_features))
        # torch.Size([391, 96])

        return all_features

    def obtain_granularball_labels_w_centers(self, inputs, centers):

        labels = assign_granular_ball_labels(inputs, centers)
            
        return labels

    def obtain_granularball_labels(self, inputs):

        # print("inputs", np.shape(inputs))
        # N, D  (391, 96)
        if np.shape(inputs)[0] > 5000:
            train_start_point_set = list(range(0, np.shape(inputs)[0]))
            batch_indices = random.sample(train_start_point_set, 5000) 
            new_inputs = inputs[batch_indices]
        else:
            new_inputs = inputs

        centers, gb_list, init_centers = func_granularball_computing(new_inputs, visual=False, labels=None)

        # print("centers", np.shape(centers))
        # print("init_centers", np.shape(init_centers))
        # centers (78, 96)
        # init_centers (19, 96)

        # LABELING
        labels = assign_granular_ball_labels(inputs, centers)
            
        return centers, labels

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)

        if self.univariate:
            data = data.squeeze()
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size] for i in range(self.sample_num)])
            X = X.unsqueeze(-1)
        else:
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size, :] for i in range(self.sample_num)])

        return X, X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index], self.gb_labels[index]

class ReconstructDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max(0, (self.data.shape[0] - window_size) // stride + 1)
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)

        if self.univariate:
            data = data.squeeze()
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size] for i in range(self.sample_num)])
            X = X.unsqueeze(-1)
        else:
            X = torch.stack([data[i * self.stride : i * self.stride + self.window_size, :] for i in range(self.sample_num)])

        return X, X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

class ForecastDataset(torch.utils.data.Dataset):
    def __init__(self, data, window_size, pred_len, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.pred_len = pred_len
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size - pred_len) // stride + 1, 0)

        # Generate samples efficiently
        self.samples, self.targets = self._generate_samples()

    def _normalize_data(self, data, epsilon=1e-8):
        """ Normalize data using mean and standard deviation. """
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        """ Generate windowed samples efficiently using vectorized slicing. """
        data = torch.tensor(self.data, dtype=torch.float32)

        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        X = torch.stack([data[i : i + self.window_size] for i in indices])
        Y = torch.stack([data[i + self.window_size : i + self.window_size + self.pred_len] for i in indices])

        return X, Y  # Inputs & targets

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.targets[index]

class TSDataset(torch.utils.data.Dataset):

    def __init__(self, X, y=None, mean=None, std=None):
        super(TSDataset, self).__init__()
        self.X = X
        self.mean = mean
        self.std = std

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.X[idx, :]

        if self.mean is not None and self.std is not None:
            sample = (sample - self.mean) / self.std
            # assert_almost_equal (0, sample.mean(), decimal=1)

        return torch.from_numpy(sample), idx


class ReconstructDataset_Moment(torch.utils.data.Dataset):
    def __init__(self, data, window_size, stride=1, normalize=True):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.data = self._normalize_data(data) if normalize else data

        self.univariate = self.data.shape[1] == 1
        self.sample_num = max((self.data.shape[0] - window_size) // stride + 1, 0)

        self.samples = self._generate_samples()
        self.input_mask = np.ones(self.window_size, dtype=np.float32)  # Fixed input mask

    def _normalize_data(self, data, epsilon=1e-8):
        mean, std = np.mean(data, axis=0), np.std(data, axis=0)
        std = np.where(std == 0, epsilon, std)  # Avoid division by zero
        return (data - mean) / std

    def _generate_samples(self):
        data = torch.tensor(self.data, dtype=torch.float32)
        indices = np.arange(0, self.sample_num * self.stride, self.stride)

        if self.univariate:
            X = torch.stack([data[i : i + self.window_size] for i in indices])
        else:
            X = torch.stack([data[i : i + self.window_size, :] for i in indices])

        return X

    def __len__(self):
        return self.sample_num

    def __getitem__(self, index):
        return self.samples[index], self.input_mask