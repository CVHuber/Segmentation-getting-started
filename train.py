from dataloader import DRIVE_Loader
from UNet import UNet
from loss import DiceBCELoss, DiceLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
import torch
import os
# 设置可用的显卡编号
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
def train():
    # 训练的epoch数
    epoch = 500
    # 数据文件夹
    img_dir = "./data/training/images"
    # 掩模文件夹
    mask_dir = "./data/training/1st_manual"
    # 网络输入图片大小
    img_size = (512, 512)
    # 创建训练loader和验证loader
    tr_loader = DataLoader(DRIVE_Loader(img_dir, mask_dir, img_size, 'train'), batch_size=4, shuffle=True,num_workers=2, pin_memory= True, drop_last=True)
    val_loader = DataLoader(DRIVE_Loader(img_dir, mask_dir, img_size, 'val'), batch_size=4, shuffle=True,num_workers=2, pin_memory= True, drop_last=True)
    # 定义损失函数
    criterion = DiceBCELoss()
    # 把网络加载到显卡
    network = UNet().cuda()
    # 定义优化器
    optimizer = Adam(network.parameters(), weight_decay=0.0001)
    best_score = 1.0
    for i in range(epoch):
        # 设置为训练模式，会更新BN和Dropout参数
        network.train()
        train_step = 0
        train_loss = 0
        val_loss = 0
        val_step = 0
        # 训练
        for batch in tr_loader:
            # 读取每个batch的数据和掩模
            imgs, mask = batch
            # 把数据加载到显卡
            imgs = imgs.cuda()
            mask = mask.cuda()
            # 把数据喂入网络，获得一个预测结果
            mask_pred = network(imgs)
            # 根据预测结果与掩模求出Loss
            loss = criterion(mask_pred, mask)
            # 统计训练loss
            train_loss += loss.item()
            train_step +=1
            # 梯度清零
            optimizer.zero_grad()
            # 通过loss求出梯度
            loss.backward()
            # 使用Adam进行梯度回传
            optimizer.step()
        # 设置为验证模式，不更新BN和Dropout参数
        network.eval()
        # 验证
        with torch.no_grad():
            for batch in val_loader:
                imgs, mask = batch
                imgs = imgs.cuda()
                mask = mask.cuda()
                # 求出评价指标，这里用的是dice
                val_loss += DiceLoss()(network(imgs), mask).item()
                val_step += 1
        # 分别求出整个epoch的训练loss以及验证指标
        train_loss /= train_step
        val_loss /= val_step
        # 如果验证指标比最优值更好，那么保存当前模型参数
        if val_loss < best_score:
            best_score = val_loss
            torch.save(network.state_dict(), "./checkpoint.pth")
        # 输出
        print(str(i), "train_loss:", train_loss, "val_dice", val_loss)

if __name__=="__main__":
    train()



