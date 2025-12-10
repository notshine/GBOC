from __future__ import division
from __future__ import print_function

import numpy as np
import math
import torch
import torch.nn.functional as F
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from torch import nn
from torch.nn import TransformerEncoder
from torch.nn import TransformerDecoder
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
import tqdm

from models.base import BaseDetector
from utils.dataset import ReconstructDataset, GB_ReconstructDataset
from utils.torch_utility import EarlyStoppingTorch, get_gpu

from granularball_computing.granularball_generation import *
from lib_instance_discrimination import *
import copy

def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GBOC_AD_Model(nn.Module):
    def __init__(self, batch_size, feats, win_size, hidden_dim, num_layers):
        super(GBOC_AD_Model, self).__init__()

        self.name = "GBOC"
        self.batch = batch_size
        self.n_feats = feats
        self.n_window = win_size
        self.n = self.n_feats * self.n_window

        self.lstm_encoder = nn.LSTM(input_size=self.n_feats, hidden_size=hidden_dim, num_layers=num_layers, batch_first=False)
        self.out_linear = nn.Linear(num_layers*hidden_dim, self.n_feats)
        
    def obtain_features(self, src):

        _, decoder_hidden = self.lstm_encoder(src)

        # decoder_hidden: output, (h_n, c_n)
        # h_n: torch.Size([3, 128, 32])  [n_layer, B, C]

        feas = decoder_hidden[0].transpose(1,0)
        B = np.shape(src)[1]
        feas = feas.reshape([B, -1]) # torch.Size([128, 96])
        return feas

    def forward(self, src, tgt):

        # print("src", np.shape(src), "tgt", np.shape(tgt))
        # src torch.Size([10, 128, 1]) tgt torch.Size([1, 128, 1])
        # T, B, C
        
        features = self.obtain_features(src)
        recon = self.out_linear(features)

        return features, recon

class GBOC(BaseDetector):
    def __init__(self,
                 win_size = 2,
                 feats = 1,
                 batch_size = 32,
                 epochs = 30,
                 patience = 10,
                 lr = 1e-4,
                 validation_size=0.2,
                 hidden_dim = 32,
                 num_layers = 3,
                 dis_mode = "ball",
                 alpha = 0.9,
                 ):
        super().__init__()

        self.__anomaly_score = None

        self.cuda = torch.cuda.is_available()
        self.device = get_device()  # Updated to use get_device() function

        self.dis_mode = dis_mode
        self.win_size = win_size
        self.batch_size = batch_size
        self.epochs = epochs
        self.feats = feats
        self.validation_size = validation_size
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.alpha = alpha         # 保存 alpha 参数

        self.model = GBOC_AD_Model(batch_size=self.batch_size, feats=self.feats, win_size=self.win_size, hidden_dim=self.hidden_dim, num_layers=self.num_layers).to(self.device)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=lr, weight_decay=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 5, 0.9)
        self.criterion = nn.MSELoss()

        self.early_stopping = EarlyStoppingTorch(None, patience=patience)

        self.discrimination_criterion = nn.CrossEntropyLoss().to(self.device)

    def collect_features(self, train_loader):

        all_features = []
        for idx, (x, _, _) in enumerate(train_loader):
            if torch.isnan(x).any() or torch.isinf(x).any():
                print("Input data contains nan or inf")
                x = torch.nan_to_num(x)

            x = x.to(self.device)
            bs = x.shape[0]
            x = x.permute(1, 0, 2)
            feas = self.model.obtain_features(x)
            all_features.append(feas)

        all_features = torch.cat(all_features, dim=0)
        
        return all_features

    def update_GB_memories(self, train_loader, epoch=0):

        # granular-ball computing
        all_features = self.collect_features(train_loader)
        features = all_features.detach().cpu().numpy()     
        centers, gb_list, init_centers = func_granularball_computing(features, visual=False, labels=None)
        tsne_visualization(features, centers, gb_list, save_path=os.path.join("plots_results", "GBOC"), save_name="epoch_{}".format(epoch))

        self.centers = centers
        print("updated centers", np.shape(centers))

    def fit(self, data):
        best_val_loss = float('inf')
        best_model_state = None
        
        # 外层循环（总epochs）
        for epoch in range(1, self.epochs + 1):
            # 每个epoch重置早停计数器
            patience_counter = 0
            epoch_best_val_loss = float('inf')
            
            tsTrain = data[:int((1-self.validation_size)*len(data))]
            tsValid = data[int((1-self.validation_size)*len(data)):]

            trainset = GB_ReconstructDataset(tsTrain, self.model, dis_mode=self.dis_mode, window_size=self.win_size)
            train_loader = DataLoader(
                dataset=trainset,
                batch_size=self.batch_size,
                shuffle=True
            )
            
            validset = GB_ReconstructDataset(tsValid, self.model, dis_mode=self.dis_mode, gb_centers=trainset.gb_centers, window_size=self.win_size)
            valid_loader = DataLoader(
                dataset=validset,
                batch_size=self.batch_size,
                shuffle=False
            )

            N_balls = np.shape(trainset.gb_centers)[0]
            self.dis_linear = nn.Linear(self.num_layers*self.hidden_dim, N_balls).to(self.device)
            
            # self.update_GB_memories(train_loader, epoch=0)

            # print(np.shape(features), np.shape(centers))
            # raise Exception(">>>>")

            centers = trainset.gb_centers
            centers = torch.tensor(centers).to(self.device)

            # 内层循环（每个epoch的微调）
            for epoch2 in range(1, 11):
                self.model.train(mode=True)
                avg_loss = 0
                loop = tqdm.tqdm(
                    enumerate(train_loader), total=len(train_loader), leave=True
                )
                for idx, (x, _, gb_y) in loop:                
                    if torch.isnan(x).any() or torch.isinf(x).any():
                        print("Input data contains nan or inf")
                        x = torch.nan_to_num(x)

                    gb_y = gb_y.to(self.device)

                    x = x.to(self.device)
                    bs = x.shape[0]
                    x = x.permute(1, 0, 2) 
                    # x: torch.Size([10, 128, 1])
                    # print(">>>>>", np.shape(x))
                    # raise Exception(">>>>")
                    
                    elem = x[-1, :, :].view(1, bs, self.feats)

                    self.optimizer.zero_grad()
                    z, x_bar = self.model(x, elem)
                    # loss = (1 / epoch) * self.criterion(z[0], elem) + (1 - 1 / epoch) * self.criterion(z[1], elem)
                    
                    # 训练阶段
                    loss1 = self.criterion(x_bar, elem)  # 重构损失
                    feas = self.model.obtain_features(x)
                    """
                    # print(np.shape(z), self.hidden_dim, N_balls)
                    """
                    # 使用交叉熵替代MSE 
                    # output = self.dis_linear(feas)
                    # loss2 = self.discrimination_criterion(output, gb_y) 
                    
                    # 用MSE替代交叉熵
                    loss2 = F.mse_loss(feas, centers[gb_y])
                    # loss = loss1
                    # loss = loss2
                    a = self.alpha  # 使用实例变量替代固定值
                    loss = a * loss1 + (1 - a) * loss2
                    # raise Exception('>>>>', np.shape(feas), np.shape())
        
                    loss.backward(retain_graph=True)

                    self.optimizer.step()
                    avg_loss += loss.cpu().item()
                    loop.set_description(f"Training Epoch [{epoch}/{self.epochs}]")
                    loop.set_postfix(loss=loss.item(), avg_loss=avg_loss / (idx + 1))

                if torch.isnan(loss):
                    print(f"Loss is nan at epoch {epoch}")
                    break

                if len(valid_loader) > 0:
                    self.model.eval()
                    avg_loss_val = 0
                    loop = tqdm.tqdm(
                        enumerate(valid_loader), total=len(valid_loader), leave=True
                    )
                    with torch.no_grad():
                        for idx, (x, _, gb_y) in loop:      

                            if torch.isnan(x).any() or torch.isinf(x).any():
                                print("Input data contains nan or inf")
                                x = torch.nan_to_num(x)

                            gb_y = gb_y.to(self.device)

                            x = x.to(self.device)
                            # x = x.unsqueeze(-1)
                            bs = x.shape[0]
                            x = x.permute(1, 0, 2)
                            elem = x[-1, :, :].view(1, bs, self.feats)

                            self.optimizer.zero_grad()
                            z, x_bar = self.model(x, elem)

                            # =====================================================
                            # loss = (1 / epoch) * self.criterion(z[0], elem) + (1 - 1 / epoch) * self.criterion(z[1], elem)
                            # 用交叉熵替代MSE
                            loss1 = self.criterion(x_bar, elem)
                            """
                            output = self.dis_linear(z)
                            loss2 = self.discrimination_criterion(output, gb_y)
                            """ 

                            # loss = loss2
                            
                            # 验证阶段
                            feas = self.model.obtain_features(x)
                            loss2 = F.mse_loss(feas, centers[gb_y])
                            # loss = loss1
                            # loss = loss2
                            loss = self.alpha * loss1 + (1 - self.alpha) * loss2
                            # loss = loss1 + loss2

                            avg_loss_val += loss.cpu().item()
                            loop.set_description(f"Validation Epoch [{epoch}/{self.epochs}]")
                            loop.set_postfix(loss=loss.item(), avg_loss_val=avg_loss_val / (idx + 1))

                # self.update_GB_memories(train_loader, epoch=epoch)

                self.scheduler.step()
                if len(valid_loader) > 0:
                    avg_loss = avg_loss_val / len(valid_loader)
                else:
                    avg_loss = avg_loss / len(train_loader)
                    
                # 早停检查，但只针对当前epoch的微调循环
                if avg_loss < epoch_best_val_loss:
                    epoch_best_val_loss = avg_loss
                    patience_counter = 0
                    # 保存当前epoch的最佳模型状态
                    epoch_best_model = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    print(f"EarlyStopping counter: {patience_counter} out of {self.early_stopping.patience}")
                    
                # 如果达到早停条件，只结束当前epoch的微调
                if patience_counter >= self.early_stopping.patience:
                    print(f"   Early stopping for epoch {epoch} <<<")
                    # 恢复到当前epoch的最佳模型
                    if epoch_best_model:
                        self.model.load_state_dict(epoch_best_model)
                    break
            
            # 更新全局最佳模型
            if epoch_best_val_loss < best_val_loss:
                best_val_loss = epoch_best_val_loss
                best_model_state = copy.deepcopy(self.model.state_dict())
                print(f"New global best model found with validation loss: {best_val_loss:.6f}")
        
        # 训练结束后，加载全局最佳模型
        if best_model_state:
            print(f"Loading best model with validation loss: {best_val_loss:.6f}")
            self.model.load_state_dict(best_model_state)

    def decision_function(self, data):

        testset = GB_ReconstructDataset(data, self.model, dis_mode=self.dis_mode, window_size=self.win_size)
        test_loader = DataLoader(
            dataset=testset,
            batch_size=self.batch_size,
            shuffle=False
        )

        self.model.eval()
        scores = []
        loop = tqdm.tqdm(enumerate(test_loader), total=len(test_loader), leave=True)

        centers = testset.gb_centers

        with torch.no_grad():
            for idx, (x, _, gb_y) in loop:

                gb_y = gb_y.to(self.device)

                x = x.to(self.device)
                bs = x.shape[0]
                x = x.permute(1, 0, 2)
                elem = x[-1, :, :].view(1, bs, self.feats)
                # breakpoint()
                # z, x_bar = self.model(x, elem)
                # loss = torch.mean(F.mse_loss(x_bar, elem, reduction="none")[0], axis=-1)
                
                # =====================================
                # use memory for detection
                feas = self.model.obtain_features(x).cpu().numpy()
                score = calculate_anomaly_scores(feas, centers)

                # print((128, 96) (947, 96))
                # torch.Size([128, 96]) (947, 96)

                # raise Exception(">>>>>>>>")

                # print("111111111111111", np.shape(score))
                scores.append(score)

        # scores = torch.cat(scores, dim=0)
        scores = np.concatenate(scores, axis=0)
        # scores = scores.numpy()
        # print(np.max(scores), np.min(scores))
        # raise Exception(">>>>")

        self.__anomaly_score = scores

        if self.__anomaly_score.shape[0] < len(data):
            self.__anomaly_score = np.array([self.__anomaly_score[0]]*math.ceil((self.win_size-1)/2) + list(self.__anomaly_score) + [self.__anomaly_score[-1]]*((self.win_size-1)//2))
        
        return self.__anomaly_score

    def anomaly_score(self) -> np.ndarray:
        return self.__anomaly_score

    def param_statistic(self, save_file):
        pass


def calculate_anomaly_scores(features, centers):
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
    squared_distances = np.maximum(squared_distances, 0)
    distances = np.sqrt(squared_distances)

    # 对于每个样本，找到距离最小的中心作为其标签
    # print(">>> distances", np.shape(distances)) (128, 947)
  
    # data = distances

    # # 第一步：按列归一化（min-max normalization）
    # # 避免除以0，加上一个很小的epsilon
    # epsilon = 1e-8
    # col_min = data.min(axis=0)
    # col_max = data.max(axis=0)
    # normalized = (data - col_min) / (col_max - col_max + epsilon)

    # # 第二步：对每一行进行排序
    # sorted_rows = np.sort(normalized, axis=1)

    # # 第三步：计算每一行相邻元素差值的最大值
    # differences = np.diff(sorted_rows, axis=1)  # 计算相邻差值
    # max_diff_per_row = np.max(differences, axis=1)  # 每行的最大差值
    # # max_diff_per_row = torch.tensor(max_diff_per_row).unsqueeze(1)
    # max_diff_per_row = np.expand_dims(max_diff_per_row, axis=1)

    # return max_diff_per_row

    distances = np.min(distances, axis=1)

    return distances
