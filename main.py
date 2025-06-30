# 导入库
# 导入torch深度学习的核心库
import torch
# 导入matplotlib绘图库
import matplotlib.pyplot as plt
# 导入sklearn的train_test_split函数
from sklearn.model_selection import train_test_split
# 导入自定义的模型和损失函数
# 明确使用了孪生网络结构和对比损失
from model import SiameseNetwork, ContrastiveLoss
# 导入自定义的数据加载器和辅助函数
from dataset import SiameseDataLoader, make_positive_indices
# 导入torch的优化器
import torch.optim as optim
# 导入自定义的训练器
from deepepochs import Trainer
# 导入自定义的数据加载函数
from data import load_data, load_test_data

# 打印环境信息
print(torch.cuda.is_available())
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print("PyTorch版本:", torch.__version__)
print("CUDA是否可用:", torch.cuda.is_available())
print("CUDA版本:", torch.version.cuda)
print("GPU数量:", torch.cuda.device_count())

# 选择设备gpu first, 然后mps, 最后cpu
device = "cuda:0" if torch.cuda.is_available() else "cpu"
device = "mps" if torch.backends.mps.is_available() else device

# 设置参数
# 数据路径
data_path = '../tinyface/train'
# 图片大小
fig_size = 32
# 验证集比例
val_ratio = 0.2
# 批量大小
batch_size = 64
# 负样本数量
neg_size = 10
# 相似度阈值
margin = 0.5

# 加载数据
# 把图片和标签读进来 gray_img=True 表示读取灰度图
images, labels = load_data(data_path, fig_size, gray_img=True)
# 创建正样本索引
positives = make_positive_indices(labels)
# 分割数据集
train_data, val_data = train_test_split(positives, test_size=val_ratio)

# 创建数据加载器
# 训练数据加载器
train_dl = SiameseDataLoader(train_data, batch_size, neg_size, images, labels, True)
# 验证数据加载器
val_dl = SiameseDataLoader(val_data, 2*batch_size, neg_size, images, labels, False)

# 创建模型
model = SiameseNetwork().to(device)
# 创建优化器
optimizer = optim.Adam(model.parameters(), lr=0.0001)
# 创建损失函数
criterion = ContrastiveLoss(margin)

# 训练模型
num_epochs = 50
# 训练损失
train_losses = []
# 验证损失
val_losses = []
# 训练准确率
train_accs = []
# 验证准确率
val_accs = []

# 训练循环
# 训练num_epochs次，训练很多轮
for epoch in range(num_epochs):
    # 告诉模型“现在是训练状态”
    model.train()
    # 用来累计这一轮所有批次的损失（loss）总和
    running_loss = 0.0
    # 用来累计这一轮总共处理了多少张图片
    total = 0
    # 用来累计这一轮总共预测对了多少张图片
    correct = 0
    # 每次从训练数据里拿出一小批图片对（left和right），还有它们的标签（是不是同一个人）
    for left_imgs, right_imgs, labels in train_dl:
        # 把这批左边的图片放到显卡
        left_imgs = left_imgs.to(device)
        # 把这批右边的图片放到显卡
        right_imgs = right_imgs.to(device)
        # 把这批标签放到显卡，并且转成浮点数，去掉多余的维度
        labels = labels.to(device).float().squeeze()
        # 清空梯度
        optimizer.zero_grad()
        # 把图片对送进模型，得到输出。这里的输出一般是两个图片的“特征向量”
        outputs = model(left_imgs, right_imgs)
        # 用损失函数（criterion）算一下模型输出和真实标签之间的差距（loss）
        loss = criterion(outputs, labels)
        # 反向传播，自动帮你算出每个参数应该怎么调整才能让loss变小
        loss.backward()
        # 用刚才算出来的梯度，更新模型的参数，让模型变得更聪明一点
        optimizer.step()
        # 把这一批的损失乘以图片数量，加到总损失里
        running_loss += loss.item() * left_imgs.size(0)
        # 把这一批的图片数量加到总数里
        total += left_imgs.size(0)
        # 计算准确率
        with torch.no_grad():
            # 算每对图片的“距离”，距离越小越像
            dists = torch.pairwise_distance(outputs[0], outputs[1])
            # 如果距离小于某个阈值（margin/2），就认为是同一个人（1），否则不是（0）
            preds = (dists < margin/2).float()  # 距离小于margin/2判为同类
            # 把预测对的数量加到总对数里
            correct += (preds == labels).sum().item()
    
    # 计算这一轮的平均损失
    avg_train_loss = running_loss / total
    # 计算这一轮的平均准确率
    train_acc = correct / total
    # 把这一轮的损失和准确率记录下来
    train_losses.append(avg_train_loss)
    train_accs.append(train_acc)

    # 验证
    # 将模型切换到“评估模式”，这会关闭如 Dropout、BatchNorm 等训练时特有的行为，保证验证时的稳定性
    model.eval()
    # 初始化验证集的累计损失、总样本数、预测正确的样本数
    val_running_loss = 0.0
    val_total = 0
    val_correct = 0
    # 在该代码块下，所有计算不会追踪梯度，节省内存和计算资源，适合验证/推理阶段
    with torch.no_grad():
        # 遍历验证集的 DataLoader，每次取出一批左图、右图和标签
        for left_imgs, right_imgs, labels in val_dl:
            # 将左图数据移动到 GPU 上
            left_imgs = left_imgs.to(device)
            # 将右图数据移动到 GPU 上
            right_imgs = right_imgs.to(device)
            # 标签转移到设备，并转换为 float 类型，去除多余的维度（squeeze）
            labels = labels.to(device).float().squeeze()
            # 前向传播，输入一对图片，输出模型结果（通常是特征向量）
            outputs = model(left_imgs, right_imgs)
            # 计算损失函数（criterion 通常是对比损失、二分类损失等）
            loss = criterion(outputs, labels)
            # 累加损失（乘以 batch size，便于后续求平均）
            val_running_loss += loss.item() * left_imgs.size(0)
            # 累加总样本数
            val_total += left_imgs.size(0)
            # 计算输出特征向量之间的欧氏距离（通常 outputs[0]、outputs[1] 分别是左、右图片的特征）
            dists = torch.pairwise_distance(outputs[0], outputs[1])
            # 以 margin/2 为阈值，距离小于阈值判为同类（1），否则为不同类（0）
            preds = (dists < margin/2).float()
            # 统计本批次中预测正确的样本数，并累加
            val_correct += (preds == labels).sum().item()
    # 计算验证集的平均损失
    avg_val_loss = val_running_loss / val_total
    # 计算验证集的平均准确率
    val_acc = val_correct / val_total
    # 记录每个 epoch 的验证损失和准确率，便于后续分析和绘图
    val_losses.append(avg_val_loss)
    val_accs.append(val_acc)

    # 打印当前 epoch 的训练损失、验证损失、训练准确率、验证准确率
    print(f"Epoch {epoch+1}/{num_epochs}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}, Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# 保存损失和准确率到文件
with open('loss_acc_log.csv', 'w') as f:
    f.write('epoch,train_loss,val_loss,train_acc,val_acc\n')
    for i, (tl, vl, ta, va) in enumerate(zip(train_losses, val_losses, train_accs, val_accs)):
        f.write(f"{i+1},{tl},{vl},{ta},{va}\n")

# 绘制损失和准确率曲线
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()
plt.subplot(1,2,2)
plt.plot(train_accs, label='Train Acc')
plt.plot(val_accs, label='Validation Acc')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()
plt.tight_layout()
plt.savefig('loss_acc_curve.png')
plt.show()

# ----- 测试

target_imgs, target_labels = load_test_data('../tinyface/test/target_faces', fig_size)
test_imgs, test_labels = load_test_data('../tinyface/test/test_faces', fig_size)

target_imgs = target_imgs.to(device)
test_imgs = test_imgs.to(device)
with torch.no_grad():
    target_embeds = model.embedding(target_imgs)
    test_embeds = model.embedding(test_imgs)


ks = [1, 3, 10]
hits = [0, 0, 0]
for embed, label in zip(test_embeds, test_labels):
    dists = torch.pairwise_distance(embed, target_embeds).cpu()
    for i, k in enumerate(ks):
        # 这里加 largest=False 是因为默认是 True，表示找最大的 k 个值，而我们需要找最小的 k 个值
        topk_idx = dists.topk(k, largest=False)[1].numpy()
        if label in target_labels[topk_idx]:
            hits[i] += 1

for k, hit in zip(ks, hits):
    print(f'hit@{k}:', hit/len(test_labels))

# 保存hit@k结果到文件
with open('hitk_log.csv', 'w') as f:
    f.write('k,hit@k\n')
    for k, hit in zip(ks, hits):
        f.write(f"{k},{hit/len(test_labels)}\n")
