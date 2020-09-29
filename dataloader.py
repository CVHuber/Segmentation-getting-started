import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
import os
from torch.utils.data import Dataset

class DRIVE_Loader(Dataset):
    def __init__(self, img_dir, mask_dir, img_size=(512, 512), mode='train'):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.mode = mode
        self.file_list = os.listdir(img_dir)
        # 以8：2的比例分割数据集作为训练集和验证集
        self.split_dataset(0.8)

    def split_dataset(self, ratio):
        # 分割训练集和验证集
        train_len = int(ratio*len(self.file_list))
        if self.mode == 'train':
            self.file_list = self.file_list[:train_len]
        else:
            self.file_list = self.file_list[train_len:]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, item):
        # 生成输入图片和掩模的文件路径
        img_file = os.path.join(self.img_dir, self.file_list[item])
        mask_file = os.path.join(self.mask_dir, self.file_list[item].replace("tif", "gif"))
        # img 和 mask 采用pillow读取，然后采用双线性插值(Bilinear)缩放成需要的尺寸
        img = np.array(Image.open(img_file).resize(self.img_size, Image.BILINEAR))
        mask = np.array(Image.open(mask_file).resize(self.img_size, Image.BILINEAR))
        # 如果读取的掩模是单通道图片，增加一个维度变成形如(224,224,1)
        if len(mask.shape)==2:
            mask = np.expand_dims(mask, axis=2)
        # HWC to CHW
        img = img.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        mask = mask / 255.0
        # 转换数据类型
        img = img.astype(np.float32)
        mask = mask.astype(np.float32)
        return torch.from_numpy(img), torch.from_numpy(mask)

if __name__ == "__main__":
    loader = DRIVE_Loader("./data/training/images", "./data/training/1st_manual", (224, 224))
    img, mask = loader.__getitem__(0)
    # 可视化展示
    plt.subplot(121)
    plt.imshow(img)
    plt.subplot(122)
    plt.imshow(mask)
    plt.show()