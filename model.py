import torch
import torch.nn as nn
import torch.nn.functional as F

# 这里定义了典型的孪生网络结构：两个输入（left_imgs, right_imgs）共用同一组网络参数，分别提取特征，最后输出两个特征向量
class SiameseNetwork(nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(8192, 128)
        self.fc2 = nn.Linear(128, 128)

    def get_embedding(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = x.view(x.size()[0], -1)
        x = F.relu(self.fc1(x))
        x = F.normalize(self.fc2(x), p=2, dim=-1)
        return x

    def forward(self, left_imgs, right_imgs):
        left_embeddings = self.get_embedding(left_imgs)
        right_embeddings = self.get_embedding(right_imgs)
        return left_embeddings, right_embeddings

    def embedding(self, images):
        return self.get_embedding(images)


# 这是孪生网络常用的对比损失函数（Contrastive Loss），用于拉近同类样本的距离，推远异类样本的距离
class ContrastiveLoss:
    """损失函数"""
    def __init__(self, margin):
        self.margin = margin

    def __call__(self, preds, label):
        left_embeddings, right_embeddings = preds
        distance = F.pairwise_distance(left_embeddings, right_embeddings)
        loss_contrastive = torch.mean(label * torch.pow(distance, 2) + (1 - label) * torch.pow(F.relu(self.margin - distance), 2))
        return loss_contrastive

