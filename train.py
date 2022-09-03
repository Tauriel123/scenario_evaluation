import torch
import torch.nn as nn
import os
from config import get_arguments
from models import ClusterNet, TAE, DbscanNet
from load_data import get_loader
from sklearn.metrics import roc_auc_score
import numpy as np
import sklearn
import pandas as pd


def pretrain_autoencoder(args, verbose=True):
    """
    function for the autoencoder pretraining
    """
    print("Pretraining autoencoder... \n")

    ## define TAE architecture
    tae = TAE(args)
    tae = tae.to(args.device)  # 把数据拷贝到device上，之后的运算都在device上进行

    ## MSE loss
    loss_ae = nn.MSELoss()  # reduction默认值为‘mean’
    ## Optimizer
    optimizer = torch.optim.Adam(tae.parameters(), lr=args.lr_ae)
    tae.train()  # 设置模块为训练模式

    for epoch in range(args.epochs_ae):
        all_loss = 0
        for batch_idx, (inputs, _) in enumerate(trainloader):

            inputs = inputs.type(torch.FloatTensor).to(args.device)
            optimizer.zero_grad()  # 把梯度清理干净，防止受之前遗留梯度的影响
            z, x_reconstr = tae(inputs)  # 把输入送到网络中，得到训练结果
            loss_mse = loss_ae(inputs.squeeze(1), x_reconstr)  # 计算当前 batch 的损失值。

            loss_mse.backward()  # 执行链式求导，计算梯度
            all_loss += loss_mse.item()
            optimizer.step()  # 更新每个可训练权重
        if verbose:
            print(
                "Pretraining autoencoder loss for epoch {} is : {}".format(
                    epoch, all_loss / (batch_idx + 1)
                )
            )

    print("Ending pretraining autoencoder. \n")
    # save weights
    torch.save(tae.state_dict(), args.path_weights_ae)


def initalize_centroids(X):
    """
    Function for the initialization of centroids.计算质心
    """
    X_tensor = torch.from_numpy(X).type(torch.FloatTensor).to(args.device)
    model.init_centroids(X_tensor)


def kl_loss_function(input, pred):
    out = input * torch.log((input) / (pred))
    return torch.mean(torch.sum(out, dim=1))


def train_ClusterNET(epoch, args, verbose):
    """
    Function for training one epoch of the DTC
    """
    model.train()
    train_loss = 0
    all_preds, all_gt = [], []
    for batch_idx, (inputs, labels) in enumerate(trainloader):
        inputs = inputs.type(torch.FloatTensor).to(args.device)
        all_gt.append(labels.cpu().detach())
        optimizer_clu.zero_grad()
        z, x_reconstr, Q, P = model(inputs)
        loss_mse = loss1(inputs.squeeze(), x_reconstr)
        loss_KL = kl_loss_function(P, Q)
        total_loss = loss_mse + loss_KL
        total_loss.backward()
        optimizer_clu.step()

        preds = torch.max(Q, dim=1)[1]
        all_preds.append(preds.cpu().detach())
        train_loss += total_loss.item()
    if verbose:
        print(
            "For epoch ",
            epoch,
            " Loss is : %.3f" % (train_loss / (batch_idx + 1)),
        )
    all_gt = torch.cat(all_gt, dim=0).numpy()
    all_preds = torch.cat(all_preds, dim=0).numpy()
    return (
        preds.detach().cpu().numpy(),
        max(
            roc_auc_score(all_gt, all_preds),
            roc_auc_score(all_gt, 1 - all_preds),
        ),
        train_loss / (batch_idx + 1),
    )


def training_function(args, verbose=True):
    """
    function for training the DTC network.
    训练整个完整模型
    """
    ## initialize clusters centroids
    initalize_centroids(X_scaled)

    ## train clustering model
    max_roc_score = 0  # 目标:最大化max_roc_score
    print("Training full model ...")
    for epoch in range(args.max_epochs):
        preds, roc_score, train_loss = train_ClusterNET(
            epoch, args, verbose=verbose
        )
        if roc_score > max_roc_score:
            max_roc_score = roc_score
            patience = 0
        else:
            patience += 1
            if patience == args.max_patience:
                break

    torch.save(model.state_dict(), args.path_weights_main)

    return max_roc_score


def training_Dbscan(args):
    epsilons, scores, models = \
        np.linspace(0.3, 1.2, 10), [], []
    # 遍历所有的半径, 训练模型, 查看得分
    for epsilon in epsilons:
        model = sklearn.DBSCAN(eps=epsilon, min_samples=5)
        model.fit(x)
        score = sklearn.silhouette_score(x, model.labels_,
                                         sample_size=len(x), metric='euclidean')
        scores.append(score)
        models.append(model)
    # 转成ndarray数组
    scores = np.array(scores)
    best_i = scores.argmax()  # 最优分数的索引
    best_eps = epsilons[best_i]
    best_sco = scores[best_i]
    print(best_eps)
    print(best_sco)
    # 获取最优模型
    best_model = models[best_i]

    # # 对输入x进行预测得到预测类别
    # pred_y = best_model.fit_predict(x)
    #
    # # 获取孤立样本, 外周样本, 核心样本
    # core_mask = np.zeros(len(x), dtype=bool)
    # # 获取核心样本的索引, 把对应位置的元素改为True
    # core_mask[best_model.core_sample_indices_] = True
    # # 孤立样本的类别标签为-1
    # offset_mask = best_model.labels_ == -1
    # # 外周样本掩码 (不是核心也不是孤立样本)
    # p_mask = ~(core_mask | offset_mask)
    return


if __name__ == "__main__":

    parser = get_arguments()
    args = parser.parse_args()
    args.path_data = args.path_data.format(args.dataset_name)
    type = 'follow'
    if type == 'follow':
        args.time_series_file = './data/follow.txt'
        args.feature_file = './data/output/follow_features.csv'
        args.label_file = './data/output/follow_label.csv'
    elif type == 'freeDrive':
        args.time_series_file = './data/freeDrive.txt'
        args.feature_file = './data/output/freeDrive_features.csv'
        args.label_file = './data/output/freeDrive_label.csv'
    elif type == 'all':
        args.time_series_file = './data/scenarios_all.txt'
        args.feature_file = './data/output/all_feature1.csv'
        args.label_file = './data/output/all_label1.csv'
    elif type == 'cutIn':
        args.time_series_file = './data/cutIn.txt'
        args.feature_file = './data/output/cutIn_feature1.csv'
        args.label_file = './data/output/cutIn_label1.csv'
    if not os.path.exists(args.path_data):
        os.makedirs(args.path_data)

    path_weights = args.path_weights.format(args.dataset_name)
    if not os.path.exists(path_weights):
        os.makedirs(path_weights)

    args.path_weights_ae = os.path.join(path_weights, "autoencoder_weight.pth")
    args.path_weights_main = os.path.join(
        path_weights, "full_model_weigths.pth"
    )

    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainloader, X_scaled = get_loader(args)
    pretrain_flag = 1
    if pretrain_flag == 1:
        pretrain_autoencoder(args)  # 预训练

    # model = ClusterNet(args)  # 构建聚类模型
    model = DbscanNet(args)
    model = model.to(args.device)
    loss1 = nn.MSELoss()
    optimizer_clu = torch.optim.SGD(
        model.parameters(), lr=args.lr_cluster, momentum=args.momentum
    )  # 优化器

    # max_roc_score = training_Dbscan(args)
    X_tensor = torch.from_numpy(X_scaled).type(torch.FloatTensor).to(args.device)
    feature_data = X_tensor.squeeze().cpu().numpy()
    feature_data = pd.DataFrame(feature_data)

    feature_data.to_csv(args.feature_file)
    result = model.DBscan_main(X_tensor)
    print(result[0])
    result = pd.DataFrame(result, columns=['clusters'])

    result.to_csv(args.label_file)
    # print("maximum roc score is {}".format(max_roc_score))
