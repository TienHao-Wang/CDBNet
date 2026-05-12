import os
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch.utils.data import DataLoader
import torch.nn.functional as F

# 导入你的模块（请确保 model4 路径正确）
from DOSOtest.MDBNet import MDBNet
from test_dataset import PipelineDataset, get_transforms
from DOSOtest.CDBNet import CDBNet

def test_and_predict_global():
    # ---------- 1. 配置参数 ----------
    CONFIG = {
        'test_data_root': r'E:\DINOv3 with CIP\CIPs\Yantai',  # 建议先测验证集对齐曲线
        'dinov3_path': 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',
        'checkpoint_path': './checkpoints_rs_cdbnet/best_model.pth',
        'save_dir': './test_global_results_CDBNet_Yantai',
        'batch_size': 32,  # 与训练保持一致
        'threshold': 0.5,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    mask_save_path = os.path.join(CONFIG['save_dir'], 'masks_tif')
    os.makedirs(mask_save_path, exist_ok=True)

    # ---------- 2. 数据加载 ----------
    test_dataset = PipelineDataset(
        image_dir=os.path.join(CONFIG['test_data_root'], 'images'),
        mask_dir=os.path.join(CONFIG['test_data_root'], 'labels'),
        transform=get_transforms(is_train=False),
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=8)

    # ---------- 3. 模型加载 ----------
    model = CDBNet(CONFIG['dinov3_path'])
    # 解决权重加载警告
    checkpoint = torch.load(CONFIG['checkpoint_path'], map_location=CONFIG['device'], weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(CONFIG['device'])
    model.eval()

    # 使用 dataset.py 中定义的属性名
    img_names = test_dataset.image_files

    # 全局计数器：用于累计整个数据集的像素统计
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_tn = 0

    print(f"载入权重 Epoch: {checkpoint.get('epoch', 'N/A')}, 训练时最高 IoU: {checkpoint.get('best_iou', 'N/A')}")
    print(f"开始全局像素评估，样本总数: {len(test_dataset)}")

    # ---------- 4. 推理与统计循环 ----------
    with torch.no_grad():
        for i, (images, masks, _) in enumerate(tqdm(test_loader)):
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])

            # 模型前向传播 (model4 输出 Mask 和 Edge)
            mask_pred, _, _ = model(images)

            # 二值化处理
            pred_mask = (torch.sigmoid(mask_pred) > CONFIG['threshold']).float()
            true_mask = masks.float()

            # 累计像素统计 (TP, FP, FN, TN)
            total_tp += (pred_mask * true_mask).sum().item()
            total_fp += (pred_mask * (1 - true_mask)).sum().item()
            total_fn += ((1 - pred_mask) * true_mask).sum().item()
            total_tn += ((1 - pred_mask) * (1 - true_mask)).sum().item()

            # 保存推理出的 8-bit TIF Mask
            preds_np = pred_mask.cpu().numpy()
            for j in range(preds_np.shape[0]):
                global_idx = i * CONFIG['batch_size'] + j
                if global_idx >= len(img_names): break

                mask_data = (preds_np[j, 0] * 255).astype(np.uint8)
                mask_pil = Image.fromarray(mask_data)

                orig_name = os.path.basename(img_names[global_idx])
                save_name = os.path.splitext(orig_name)[0] + ".tif"
                mask_pil.save(os.path.join(mask_save_path, save_name))

    # ---------- 5. 全局指标计算 ----------
    # Accuracy: (TP + TN) / Total
    accuracy = (total_tp + total_tn) / (total_tp + total_tn + total_fp + total_fn + 1e-7)
    # Precision: TP / (TP + FP)
    precision = total_tp / (total_tp + total_fp + 1e-7)
    # Recall: TP / (TP + FN)
    recall = total_tp / (total_tp + total_fn + 1e-7)
    # F1-Score: 2 * P * R / (P + R)
    f1_score = 2 * precision * recall / (precision + recall + 1e-7)
    # Foreground IoU: TP / (TP + FP + FN)
    iou_fg = total_tp / (total_tp + total_fp + total_fn + 1e-7)
    # Background IoU: TN / (TN + FP + FN)
    iou_bg = total_tn / (total_tn + total_fp + total_fn + 1e-7)
    # mIoU: Mean of FG and BG IoU
    miou = (iou_fg + iou_bg) / 2

    # ---------- 6. 报告打印 ----------
    print("\n" + "═" * 50)
    print("       DINO-PipeNet 细长管线提取 - 全局累计评估报告")
    print("═" * 50)
    print(f"像素准确率 (Accuracy):     {accuracy:.4f}")
    print(f"精确率 (Precision):        {precision:.4f} (误报率: {1 - precision:.4f})")
    print(f"召回率 (Recall):           {recall:.4f} (漏检率: {1 - recall:.4f})")
    print(f"F1 综合得分 (F1-Score):     {f1_score:.4f}")
    print("-" * 50)
    print(f"前景 IoU (Pipeline Class): {iou_fg:.4f}  <-- 核心硬核指标")
    print(f"背景 IoU (Environ Class):  {iou_bg:.4f}")
    print(f"平均交并比 (mIoU):         {miou:.4f}")
    print("═" * 50)
    print(f"预测 Mask 已存入: {mask_save_path}")


if __name__ == "__main__":
    test_and_predict_global()
