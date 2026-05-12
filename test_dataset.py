import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PipelineDataset(Dataset):
    """
    输油管道数据集
    - 影像: 512x512 RGB
    - 标签: 512x512 TIFF (255=前景, 0=背景)
    """

    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment

        # 获取所有图像文件名
        self.image_files = sorted([f for f in os.listdir(image_dir)
                                   if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取影像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # 读取标签
        mask_name = self.image_files[idx].replace('.jpg', '.png')#.replace('.jpg', '.tif').replace('.png', '.tif')
        mask_path = os.path.join(self.mask_dir, mask_name)
        mask = Image.open(mask_path)

        # 转换为 numpy
        image = np.array(image)
        mask = np.array(mask)

        # 数据增强
        if self.augment:
            image, mask = self.augment_data(image, mask)

        # 标签归一化: 255 -> 1
        mask = (mask > 127).astype(np.float32)

        # 转换为 Tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = torch.from_numpy(mask).unsqueeze(0)  # [1, H, W]

        # 计算边缘 (用于辅助监督)
        edge = self.extract_edge(mask.squeeze().numpy())
        edge = torch.from_numpy(edge).unsqueeze(0)

        return image, mask, edge

    def augment_data(self, image, mask):
        """数据增强"""
        # 随机水平翻转
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        # 随机垂直翻转
        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        # 随机旋转 (90度的倍数)
        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        return image, mask

    @staticmethod
    def extract_edge(mask, kernel_size=5):
        """提取边缘"""
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)
        edge = (dilated - eroded) / 255.0
        return edge.astype(np.float32)


def get_transforms(is_train=True):
    """获取数据预处理"""
    if is_train:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
