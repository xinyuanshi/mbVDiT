import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from torch import optim
from torch.autograd import Variable
import datetime
import random
import time
import logging
LOGGER = logging.getLogger()
from tqdm import tqdm
from sklearn.preprocessing import MinMaxScaler
import os
#from TransVCOX.pre_train.vae_main import *
import scprep as scp
import scanpy as sc
from sklearn.model_selection import train_test_split

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
seed = random.randint(2000, 5000)
# 设置PyTorch的随机种子
torch.manual_seed(seed)
random.seed(seed)
# （可选）设置CUDA的随机种子，如果你使用GPU
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# 设置NumPy的随机种子
np.random.seed(seed)


class MyDataset(Dataset):
    def __init__(self, features):
        self.features = features.astype(float)
        self.features = torch.Tensor(features)

    def __getitem__(self, index):
        self.sample_features = self.features[index]
        return self.sample_features

    def __len__(self):
        return len(self.features)


def dataset():
    # 读取CSV文件
    #df = sc.read("D:/TC.csv", sep='\t', first_column_names=True)
    df = sc.read("D:/PyCharm/Py_Projects/DiTT/datasets/data_ucec/st/data_ucec_UCEC_1695.h5ad", sep='\t')
    #sc.pp.normalize_total(df, target_sum=1e5)  # 1e4
    #sc.pp.log1p(df)
    df = df.X
    # df = df.transpose()
    # 划分数据集
    train_data, test_data = train_test_split(df, test_size=0.2)
    train_features = train_data
    test_features = test_data

    print('Origin:')
    print(train_features)

    # train_features = np.transpose(train_features)
    # test_features = np.transpose(test_features)

    print('Tr:')
    print(train_features)

    # train_features = scaler.fit_transform(train_features)
    # test_features = scaler.transform(test_features)

    # train_features = pd.DataFrame(train_features)
    print('stand:')
    print(train_features)
    # train_features = np.transpose(train_features)
    # test_features = np.transpose(test_features)

    features = [train_features, test_features]
    trainset = MyDataset(features[0])
    testset = MyDataset(features[1])
    train_loader = DataLoader(trainset, batch_size=512, shuffle=True)
    test_loader = DataLoader(testset, batch_size=512, shuffle=False)
    return train_loader, test_loader


class reparametrize(nn.Module):
    def __init__(self):
        super(reparametrize, self).__init__()

    def forward(self, z_mean, z_log_var):
        epsilon = torch.randn(z_mean.shape)
        epsilon = epsilon.to(device)
        return z_mean + (z_log_var / 2).exp() * epsilon

#256, 100, 0.5
class VaeEncoder(nn.Module):
    def __init__(self):
        super(VaeEncoder, self).__init__()
        self.Dense = nn.Linear(1695, 512)
        self.z_mean = nn.Linear(512, 128)
        self.z_log_var = nn.Linear(512, 128)
        self.dropout = nn.Dropout(p=0.1)
        self.sample = reparametrize()

    def forward(self, x):
        o = torch.nn.functional.relu(self.Dense(x))
        o = self.dropout(o)
        z_mean = self.z_mean(o)
        z_log_var = self.z_log_var(o)
        o = self.sample(z_mean, z_log_var)
        return o, z_mean, z_log_var


class VaeDecoder(nn.Module):
    def __init__(self):
        super(VaeDecoder, self).__init__()
        self.Dense = nn.Linear(128, 512)
        self.out = nn.Linear(512, 1695)
        self.dropout = nn.Dropout(p=0.1)
        #self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, z):
        o = nn.functional.relu(self.Dense(z))
        o = self.dropout(o)
        o = self.out(o)
        return self.relu(o)


class Vae(nn.Module):
    def __init__(self):
        super(Vae, self).__init__()
        self.encoder = VaeEncoder()
        self.decoder = VaeDecoder()

    def forward(self, x):
        o, mean, var = self.encoder(x)
        return self.decoder(o), mean, var, o


model = Vae()
model = model.to(device)
print(model)

data_set = dataset()


reconstruction_function = nn.MSELoss(reduction='mean')

optimizer = optim.Adam(model.parameters(), lr=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', factor=0.1, patience=5, verbose=False)

def loss_function(recon_x, x, mu, logvar):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    BCE = reconstruction_function(recon_x, x)  # mse loss
    # loss = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    # KL divergence
    return BCE + KLD

def pre_train_model_ready(num_epochs):
    print('VAE.....................................................')

    model.load_state_dict(torch.load('D:/ucec_vae_parameters_test.pth'))

    for epoch in range(num_epochs):
        train_loss = 0
        for batch_idx, data in enumerate(data_set[0]):
            model.train()
            starttime = datetime.datetime.now()
            mic_data = data
            #mic_data = Variable(mic_data)
            mic_data = (mic_data.to(device) if torch.cuda.is_available() else mic_data)
            recon_batch, mu, logvar, _ = model(mic_data)
            loss = loss_function(recon_batch, mic_data, mu, logvar)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step(loss)

            train_loss += loss.item()
            '''
            if batch_idx % 100 == 0:
                endtime = datetime.datetime.now()
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} time:{:.2f}s'.format(
                    epoch,
                    batch_idx * len(mic_data),
                    len(train_loader.dataset),
                    100. * batch_idx / len(train_loader),
                    loss.item() / len(mic_data),
                    (endtime-starttime).seconds))
            '''
        print('====> Epoch: {} Average loss: {:.4f}'.format(
            epoch, train_loss / len(data_set[0].dataset)))
        #print(len(train_loader.dataset))

pre_train_model_ready(300)

model.eval()

with torch.no_grad():
    for batch_idx, data in enumerate(data_set[1]):
        mic_data = data
        # mic_data = Variable(mic_data)
        mic_data = (mic_data.to(device) if torch.cuda.is_available() else mic_data)
        recon_batch, mu, logvar, o = model(mic_data)
        print()


def rmse(matrix1, matrix2):
    # 计算两个矩阵之间的差异
    diff = matrix1 - matrix2

    # 将差异平方
    squared_diff = np.square(diff)

    # 对平方差取平均
    mean_squared_diff = np.mean(squared_diff)

    # 对平均平方差取平方根
    rmse_value = np.sqrt(mean_squared_diff)

    return rmse_value

m1 = np.array(data_set[1].dataset.features)
m2 = np.array(recon_batch.cpu())
rmse_value = rmse(m1, m2)


def mae(matrix1, matrix2):
    # 计算两个矩阵之间的差异
    diff = matrix1 - matrix2

    # 取差异的绝对值
    abs_diff = np.abs(diff)

    # 对绝对值差异取平均
    mae_value = np.mean(abs_diff)

    return mae_value

mae_value = mae(m1, m2)
print(rmse_value)

file_path = 'D:/ucec_vae_parameters_tune_test.pth'
torch.save(model.state_dict(), file_path)