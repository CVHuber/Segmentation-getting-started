import torch
from PIL import Image
import numpy as np
import os
from UNet import UNet
# 设置使用的显卡编号
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
def test(test_dir, weight_path, outputs_dir, img_size=(512, 512)):
    """
    :param test_dir:需要预测的数据的文件夹
    :param weight_path: 权重文件路径
    :param outputs_dir: 输出文件夹
    :param img_size: 图片大小
    :return:
    """
    # 定义网络结构，并且加载到显卡
    network = UNet().cuda()
    # 加载权重文件（训练好的网络）
    network.load_state_dict(torch.load(weight_path))
    # 获取测试文件夹的文件
    file_list = os.listdir(test_dir)
    for f in file_list:
        # 读取图片并完成缩放
        img = np.array(Image.open(os.path.join(test_dir, f)).resize(img_size, Image.BILINEAR))
        # 增加batch维度
        img = np.expand_dims(img, axis=0)
        # 更改通道顺序（BHWC->BCHW）
        img = img.transpose((0, 3, 1, 2))
        # 转为浮点类型
        img = img.astype(np.float32)
        # 预测结果并且从显存转移到内存中
        pred = network(torch.from_numpy(img).cuda()).clone().cpu().detach().numpy()
        # 二值化操作
        pred[pred >= 0.5] = 1
        pred[pred < 0.5] = 0
        # 保存结果到输出文件夹
        Image.fromarray(pred[0, 0, :, :]).save(os.path.join(outputs_dir, f))


if __name__ == "__main__":
    test("./data/test/images", "./checkpoint/best.pth", "./data/test/outputs")