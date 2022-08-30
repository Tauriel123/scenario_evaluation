import numpy as np
from sklearn.preprocessing import LabelEncoder
from tslearn.preprocessing import TimeSeriesScalerMeanVariance
from tslearn.datasets import UCR_UEA_datasets
import torch
from torch.utils.data import Dataset, DataLoader
from tslearn.utils import save_time_series_txt, load_time_series_txt

"""
关于CustomDataset和DataLoader：
我们一般使用一个for循环（或多层的）来训练神经网络，每一次迭代，加载一个batch的数据，神经网络前向反向传播各一次并更新一次参数。
而这个过程中加载一个batch的数据这一步需要使用一个torch.utils.data.DataLoader对象，
并且DataLoader是一个基于某个dataset的iterable，这个iterable每次从dataset中基于某种采样原则取出一个batch的数据。
也可以这样说：Torch中可以创建一个torch.utils.data.Dataset对象，并与torch.utils.data.DataLoader一起使用，
在训练模型时不断为模型提供数据。
"""

def load_ucr(args, scale):
    """
    Load ucr dataset.
    Taken from https://github.com/FlorentF9/DeepTemporalClustering/blob/4f70d6b24722bd9f8331502d9cae00d35686a4d2/datasets.py#L18
    """
    # X_train, y_train, X_test, y_test = UCR_UEA_datasets().load_dataset(dataset)
    # X = np.concatenate((X_train, X_test))
    # y = np.concatenate((y_train, y_test))
    # print(np.shape(X_train))
    # print(np.shape(y_train))
    X = load_time_series_txt(args.time_series_file)  # './data/freeDrive.txt'
    print(len(X))

    # y = np.random.rand((np.shape(X)[0],))
    # print(np.shape(X))
    # print(np.shape(y))
    # y = np.array([int(i/10) for i in range(np.shape(X)[0])])
    Y1 = [1 for i in range(200)]
    Y2 = [0 for i in range(np.shape(X)[0] - 200)]
    y = np.array(Y1 + Y2)
    if args.dataset_name == "HandMovementDirection":  # this one has special labels
        y = [yy[0] for yy in y]
    y = LabelEncoder().fit_transform(
        y
    )  # sometimes labels are strings or start from 1
    assert y.min() == 0  # assert labels are integers and start from 0
    # preprocess data (standardization)
    if scale:
        X = TimeSeriesScalerMeanVariance().fit_transform(X)
    return X, y


def load_data(args, scale=True):
    """
    args :
        dataset_name : a string representing the dataset name.
        all_ucr_datasets : a list of all ucr datasets present in tslearn UCR_UEA_datasets
        scale : a boolean that represents whether to scale the time series or not.
    return :
        X : time series , scaled or not.
        y : the labels ( binary in our case) . s
    """
    # if dataset_name in all_ucr_datasets:
    #     return load_ucr(dataset_name, scale)
    # else:
    #     print(
    #         "Dataset {} not available! Available datasets are UCR/UEA univariate and multivariate datasets.".format(
    #             dataset_name
    #         )
    #     )
    #     exit(0)
    return load_ucr(args, scale)


class CustomDataset(Dataset):
    def __init__(self, time_series, labels):
        """
        This class creates a torch dataset.

        """
        self.time_series = time_series
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        time_serie = torch.tensor(self.time_series[idx]).squeeze().unsqueeze(0) #取一行
        label = torch.tensor(self.labels[idx])

        return (time_serie, label)


def get_loader(args):
    args.dataset_name = 'NGSIM'
    X_scaled, y = load_data(args)
    # create dataset
    # print(X_scaled)
    # print(np.shape(X_scaled))

    trainset = CustomDataset(X_scaled, y)
    # print(trainset)

    ## create dataloader
    trainloader = DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True, num_workers=2
    )
    args.serie_size = X_scaled.shape[1]
    return trainloader, X_scaled #标签和样本都是样本本身

    # trainloader = load_time_series_txt(
    #     'E:/学位论文/code/Deep-temporal-clustering-main-pytorch/Deep-temporal-clustering-main/data/NGSIM/scenarios.txt')
    # trainloader = load_time_series_txt(
    #     'E:/学位论文/code/Deep-temporal-clustering-main-pytorch/Deep-temporal-clustering-main/data/NGSIM/scenarios_without.txt')
    # # trainloader=trainloader.reshape(-1,1,30)
    # print(np.shape(trainloader))
    #
    #
    # args.serie_size = trainloader.shape[1]
    #
    # # create dataset
    # y = np.array([1 for i in range(np.shape(trainloader)[0])])
    # trainloader = TimeSeriesScalerMeanVariance().fit_transform(trainloader)
    # print(np.shape(trainloader))
    # trainset = CustomDataset(trainloader, y)
    # trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    # print(type(trainloader))
