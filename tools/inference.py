import torch
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

from model4 import build_model


def predict_single_image(model, image_path, device, threshold=0.5):
    """对单张图像进行预测"""
    model.eval()

    # 读取图像
    image = Image.open(image_path).convert('RGB')
    original_size = image.size

    # 预处理
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    input_tensor = transform(image).unsqueeze(0).to(device)

    # 推理
    with torch.no_grad():
        mask_pred, _ = model(input_tensor)
        mask_pred = torch.sigmoid(mask_pred)

    # 后处理
    mask = mask_pred.squeeze().cpu().numpy()
    mask = (mask > threshold).astype(np.uint8) * 255

    # 调整回原始尺寸
    mask = cv2.resize(mask, original_size, interpolation=cv2.INTER_NEAREST)

    return mask


def visualize_result(image_path, mask, save_path=None):
    """可视化结果"""
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 创建彩色掩码
    colored_mask = np.zeros_like(image)
    colored_mask[mask > 0] = [255, 0, 0]  # 红色

    # 叠加
    overlay = cv2.addWeighted(image, 0.7, colored_mask, 0.3, 0)

    # 显示
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    axes[1].imshow(mask, cmap='gray')
    axes[1].set_title('Predicted Mask')
    axes[1].axis('off')

    axes[2].imshow(overlay)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # 配置
    model_path = './checkpoints6/best_model.pth'
    image_path = r"E:\DINOv3 with CIP\paperchart\Generalizeimages\N_1280_3200.tif"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 加载模型
    model = build_model(dinov3_path='dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)

    print(f"模型加载成功 (Val IoU: {checkpoint.get('val_iou', 'N/A')})")

    # 预测
    mask = predict_single_image(model, image_path, device)

    # 可视化
    visualize_result(image_path, mask, save_path='result3.png')


if __name__ == '__main__':
    main()
