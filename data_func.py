
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder

matplotlib.use('Agg')
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_dataset(args):

    # if args.dataset == 'TSBAD':
        
    folder = os.path.join(args.data_direc, args.filename+".csv")

    # Loading Data
    df = pd.read_csv(folder).dropna()
    data = df.iloc[:, 0:-1].values.astype(float)
    labels = df['Label'].astype(int).to_numpy()

    # Univariate
    print(np.shape(data),np.shape(labels))
    # (4031, 1) (4031,)

    # Multivariate
    # print(np.shape(data),np.shape(labels))
    # (46660, 18) (46660,)

    plt.figure(figsize=(12,6))
    if "TSB-AD-U" in args.data_direc:
        index = 0
    else:
        index = 0
    _n = np.shape(labels)[0]
    x = np.array([[i] for i in range(_n)])[:,0]

    min_ = np.min(data[:,index])
    y1 = np.array([min_ for i in range(_n)])
    max_ = np.max(data[:,index])
    y2 = np.array([max_*labels[i] for i in range(_n)])
    plt.fill_between(x, y1, y2, where=y2, alpha=0.2, color='red')

    plt.plot(data[:,index])
    plt.tight_layout()
    plt.savefig(os.path.join("plots_data", "{}.png".format(args.filename)))

    return data, labels


