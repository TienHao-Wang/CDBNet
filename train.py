import os
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from tqdm import tqdm
import numpy as np
import matplotlib
import torch.nn as nn
import torch.nn.functional as F

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

from CDBNet import CDBNet
from train_dataset import PipelineDataset, get_transforms

# 设置中文字体支持
rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False


class DiceLoss(torch.nn.Module):
    """Dice Loss"""

    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)

        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        return 1 - dice

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.5, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # 预测正确的概率
        F_loss = self.alpha * (1 - pt)**self.gamma * BCE_loss
        return F_loss.mean()

class BoundaryLoss(nn.Module):
    """边缘损失 - 强制模型关注管道边缘"""

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, pred, target):
        return self.bce(pred, target)


class CombinedLoss(nn.Module):
    """组合损失函数"""

    def __init__(self, alpha=1.0, beta=0.5, gamma=0.5, sigma=1):
        super().__init__()
        self.alpha = alpha  # Dice Loss 权重
        self.beta = beta  # BCE Loss 权重
        self.gamma = gamma  # Boundary Loss 权重
        self.sigma = sigma

        self.dice_loss = DiceLoss()
        #self.bce_loss = nn.BCEWithLogitsLoss()
        self.focal_loss = FocalLoss()
        self.boundary_loss = BoundaryLoss()

    def forward(self, mask_pred, mask_target, edge_pred, edge_target, skeleton_pred, skeleton_target):
        # 主分割损失
        #coarse = self.dice_loss(coarse_pred, coarse_target)
        #bce = self.bce_loss(mask_pred, mask_target)
        focal = self.focal_loss(mask_pred,mask_target)

        # 边缘损失
        boundary = self.boundary_loss(edge_pred, edge_target)
        skeleton = self.boundary_loss(skeleton_pred, skeleton_target)

        total_loss = self.gamma * boundary + self.sigma * focal + self.beta * skeleton

        return total_loss, {
            #'coarse': coarse.item(),
            'skeleton': skeleton.item(),
            'boundary': boundary.item(),
            'focal': focal.item()
        }


def calculate_iou(pred, target, threshold=0.5):
    """计算 IoU"""
    pred = (pred > threshold).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def train_one_epoch(model, dataloader, criterion, optimizer, scheduler, device, epoch):
    """训练一个 epoch"""
    model.train()

    running_loss = 0.0
    running_iou = 0.0

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    for images, masks, edge, skeleton in pbar:
        images = images.to(device)
        masks = masks.to(device)
        edge = edge.to(device)
        skeleton = skeleton.to(device)

        # 前向传播
        pred, edge_pred, skeleton_pred = model(images)

        # 计算损失
        loss, loss_dict = criterion(pred, masks, edge_pred, edge, skeleton_pred, skeleton)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # 统计
        running_loss += loss.item()
        running_iou += calculate_iou(pred, masks)

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'egde': f'{loss_dict["boundary"]:.4f}',
            'skeleton': f'{loss_dict["skeleton"]:.4f}',
            'iou': f'{running_iou / (pbar.n + 1):.4f}'
        })

    return running_loss / len(dataloader), running_iou / len(dataloader)


def validate(model, dataloader, criterion, device):
    """验证"""
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for images, masks, edges, skeletes in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            masks = masks.to(device)
            edges = edges.to(device)
            skeletes = skeletes.to(device)

            pred,edge_pred, skelete_pred = model(images)
            loss, _ = criterion(pred, masks, edge_pred, edges, skelete_pred, skeletes)

            running_loss += loss.item()
            running_iou += calculate_iou(pred, masks)

    return running_loss / len(dataloader), running_iou / len(dataloader)


def plot_training_curves(history, save_path):
    """绘制训练曲线"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    # 左侧 y 轴：损失
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold', color='tab:red')

    line1 = ax1.plot(epochs, history['train_loss'],
                     color='#FF6B6B', linewidth=2, marker='o',
                     markersize=4, label='Train Loss', alpha=0.8)

    line2 = ax1.plot(epochs, history['val_loss'],
                     color='#FF0000', linewidth=2, marker='s',
                     markersize=4, label='Val Loss', alpha=0.8)

    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, alpha=0.3, linestyle='--')

    # 右侧 y 轴：IoU
    ax2 = ax1.twinx()
    ax2.set_ylabel('IoU', fontsize=12, fontweight='bold', color='tab:blue')

    line3 = ax2.plot(epochs, history['train_iou'],
                     color='#4ECDC4', linewidth=2, marker='^',
                     markersize=4, label='Train IoU', alpha=0.8)

    line4 = ax2.plot(epochs, history['val_iou'],
                     color='#0066CC', linewidth=2, marker='D',
                     markersize=4, label='Val IoU', alpha=0.8)

    ax2.tick_params(axis='y', labelcolor='tab:blue')

    # 标注最佳点
    best_epoch = np.argmax(history['val_iou']) + 1
    best_iou = max(history['val_iou'])

    ax2.scatter([best_epoch], [best_iou],
                color='gold', s=200, marker='*',
                edgecolors='black', linewidths=2,
                zorder=5, label=f'Best (Epoch {best_epoch})')

    ax2.annotate(f'Best IoU: {best_iou:.4f}\nEpoch: {best_epoch}',
                 xy=(best_epoch, best_iou),
                 xytext=(10, -30), textcoords='offset points',
                 bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7),
                 arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                 color='black', lw=1.5),
                 fontsize=10, fontweight='bold')

    # 图例
    lines = line1 + line2 + line3 + line4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=10, framealpha=0.9)

    plt.title('Training and Validation Curves',
              fontsize=14, fontweight='bold', pad=20)
    fig.tight_layout()

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ 训练曲线已保存至: {save_path}")
    plt.close()


def main():
    # 配置
    CONFIG = {
        'data_root': r'E:\road extraction\deepglobe-road-dataset\DeepGlobe',
        'dinov3_path': 'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth',  # 修改为你的 DINOv3 权重路径
        'batch_size': 16,
        'num_epochs': 100,
        'lr': 0.0005,
        'weight_decay': 1e-4,
        'num_workers': 8,
        'save_dir': './checkpoints_rs_cdbnet_deepglobe'
    }

    os.makedirs(CONFIG['save_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    train_dataset = PipelineDataset(
        image_dir=os.path.join(CONFIG['data_root'], 'train/images'),
        mask_dir=os.path.join(CONFIG['data_root'], 'train/labels'),
        transform=get_transforms(is_train=True),
        augment=True
    )

    val_dataset = PipelineDataset(
        image_dir=os.path.join(CONFIG['data_root'], 'test/images'),
        mask_dir=os.path.join(CONFIG['data_root'], 'test/labels'),
        transform=get_transforms(is_train=False),
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG['batch_size'],
        shuffle=False,
        num_workers=CONFIG['num_workers'],
        pin_memory=True
    )

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")

    # 模型构建
    model = CDBNet(CONFIG['dinov3_path'])
    model = model.to(device)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"可训练参数: {trainable_params:,} / {total_params:,}")

    # 优化器与损失
    optimizer = AdamW(
        model.parameters(),
        lr=CONFIG['lr'],
        weight_decay=CONFIG['weight_decay']
    )

    scheduler = OneCycleLR(
        optimizer,
        max_lr=CONFIG['lr'],
        steps_per_epoch=len(train_loader),
        epochs=CONFIG['num_epochs']
    )

    criterion = CombinedLoss()

    # 训练历史
    history = {
        'train_loss': [],
        'train_iou': [],
        'val_loss': [],
        'val_iou': []
    }

    best_iou = 0.0
    best_epoch = 0

    # 训练循环
    for epoch in range(1, CONFIG['num_epochs'] + 1):
        train_loss, train_iou = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, device, epoch
        )

        val_loss, val_iou = validate(model, val_loader, criterion, device)

        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_iou'].append(train_iou)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)

        print(f"\nEpoch {epoch}/{CONFIG['num_epochs']}")
        print(f"Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val IoU: {val_iou:.4f}")
        print(f"LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 保存最佳模型
        if val_iou > best_iou:
            best_iou = val_iou
            best_epoch = epoch

            best_model_path = os.path.join(CONFIG['save_dir'], 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': train_loss,
                'train_iou': train_iou,
                'val_loss': val_loss,
                'val_iou': val_iou,
                'best_iou': best_iou,
                'history': history
            }, best_model_path)

            print(f"✓ 保存最佳模型 (Epoch {epoch}, Val IoU: {val_iou:.4f})")

        print(f"当前最佳: Epoch {best_epoch}, IoU: {best_iou:.4f}\n")

        # 每 5 个 epoch 绘制曲线
        if epoch % 5 == 0 or epoch == CONFIG['num_epochs']:
            plot_path = os.path.join(CONFIG['save_dir'], f'training_curve_epoch_{epoch}.png')
            plot_training_curves(history, plot_path)

    # 保存最后模型
    last_model_path = os.path.join(CONFIG['save_dir'], 'last_model.pth')
    torch.save({
        'epoch': CONFIG['num_epochs'],
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'train_iou': train_iou,
        'val_loss': val_loss,
        'val_iou': val_iou,
        'best_iou': best_iou,
        'best_epoch': best_epoch,
        'history': history
    }, last_model_path)

    # 绘制最终曲线
    final_plot_path = os.path.join(CONFIG['save_dir'], 'final_training_curve.png')
    plot_training_curves(history, final_plot_path)

    print(f"\n{'=' * 60}")
    print(f"训练完成！")
    print(f"最佳模型: Epoch {best_epoch}, Val IoU: {best_iou:.4f}")
    print(f"最后模型: Epoch {CONFIG['num_epochs']}, Val IoU: {val_iou:.4f}")
    print(f"模型保存路径: {CONFIG['save_dir']}")
    print(f"训练曲线: {final_plot_path}")
    print(f"{'=' * 60}\n")


if __name__ == '__main__':
    main()
