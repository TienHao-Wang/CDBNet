import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class PipelineDataset(Dataset):
    """
    化工管线数据集
    - 影像: RGB
    - 标签: 255=前景, 0=背景
    - 返回: image, mask, edge, skeleton
    """

    def __init__(self, image_dir, mask_dir, transform=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.augment = augment

        self.image_files = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 读取影像
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # 读取标签
        base_name = os.path.splitext(self.image_files[idx])[0]
        mask_path = os.path.join(self.mask_dir, base_name + '.tif')

        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base_name + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_dir, base_name + '.jpg')
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask file not found for image: {self.image_files[idx]}")

        mask = Image.open(mask_path)

        # 转 numpy
        image = np.array(image)
        mask = np.array(mask)

        # 如果 mask 是多通道，只取一个通道
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        # 数据增强：必须同时作用于 image 和 mask
        if self.augment:
            image, mask = self.augment_data(image, mask)

        # 标签二值化: 255 -> 1
        mask = (mask > 127).astype(np.float32)

        # 根据增强后的 mask 生成 edge 和 skeleton
        edge = self.extract_edge(mask, kernel_size=5)
        skeleton = self.extract_skeleton(mask)

        # 转 Tensor
        if self.transform:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask = torch.from_numpy(mask).unsqueeze(0).float()          # [1, H, W]
        edge = torch.from_numpy(edge).unsqueeze(0).float()          # [1, H, W]
        skeleton = torch.from_numpy(skeleton).unsqueeze(0).float()  # [1, H, W]

        return image, mask, edge, skeleton

    def augment_data(self, image, mask):
        """数据增强"""
        if np.random.rand() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if np.random.rand() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        k = np.random.randint(0, 4)
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        return image, mask

    @staticmethod
    def extract_edge(mask, kernel_size=5):
        """
        提取边缘标签。
        mask: H×W, 0/1
        return: H×W, 0/1
        """
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (kernel_size, kernel_size)
        )

        dilated = cv2.dilate(mask_uint8, kernel, iterations=1)
        eroded = cv2.erode(mask_uint8, kernel, iterations=1)

        edge = (dilated - eroded) > 0
        return edge.astype(np.float32)

    @staticmethod
    def extract_skeleton(mask):
        """
        提取 skeleton 标签。
        mask: H×W, 0/1
        return: H×W, 0/1
        """
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255

        # 优先使用 OpenCV contrib 的 thinning
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skeleton = cv2.ximgproc.thinning(
                mask_uint8,
                thinningType=cv2.ximgproc.THINNING_ZHANGSUEN
            )
            skeleton = (skeleton > 0).astype(np.float32)
            return skeleton

        # 如果没有 cv2.ximgproc，则使用形态学骨架提取作为 fallback
        return PipelineDataset.morphological_skeleton(mask_uint8)

    @staticmethod
    def morphological_skeleton(mask_uint8):
        """
        形态学骨架提取，不依赖 skimage / opencv-contrib。
        mask_uint8: H×W, 0/255
        return: H×W, 0/1
        """
        img = (mask_uint8 > 0).astype(np.uint8) * 255
        skeleton = np.zeros_like(img, dtype=np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            eroded = cv2.erode(img, kernel)
            opened = cv2.dilate(eroded, kernel)
            temp = cv2.subtract(img, opened)
            skeleton = cv2.bitwise_or(skeleton, temp)
            img = eroded.copy()

            if cv2.countNonZero(img) == 0:
                break

        skeleton = (skeleton > 0).astype(np.float32)
        return skeleton


def get_transforms(is_train=True):
    """获取数据预处理"""
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
