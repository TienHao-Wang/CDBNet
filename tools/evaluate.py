import os
os.environ['OPENCV_LOG_LEVEL'] = 'SILENT'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'
import cv2
import numpy as np
from tqdm import tqdm
from skimage.measure import label as sk_label
from skimage.morphology import skeletonize
import warnings
import networkx as nx
from scipy.spatial import cKDTree

# 忽略 TIFF 地理信息警告
warnings.filterwarnings('ignore', category=UserWarning)


class PipelineEvaluator:
    def __init__(self, label_dir="test/labels", pred_dir="results"):
        self.label_dir = label_dir
        self.pred_dir = pred_dir
        self.reset()

    def reset(self):
        """重置累计指标"""
        # 全局累计混淆矩阵
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.TN = 0

        # 连通性指标（仅对有前景的图计算）
        self.connectivity_scores = []
        self.completeness_scores = []

        # APLS 指标
        self.apls_scores = []

        self.total_images = 0
        self.foreground_images = 0  # 有前景的图像数量

    def load_image(self, path):
        """加载图像并二值化"""
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"无法读取图像: {path}")
        # 二值化：前景=1，背景=0
        binary = (img > 127).astype(np.uint8)
        return binary

    def accumulate_confusion_matrix(self, gt_label, pred_label):
        """累计混淆矩阵（全局统计）"""
        self.TP += np.sum((gt_label == 1) & (pred_label == 1))
        self.FP += np.sum((gt_label == 0) & (pred_label == 1))
        self.FN += np.sum((gt_label == 1) & (pred_label == 0))
        self.TN += np.sum((gt_label == 0) & (pred_label == 0))

    def skeleton_to_graph(self, skeleton):
        """将骨架图转换为图结构"""
        # 获取骨架点
        points = np.argwhere(skeleton > 0)

        if len(points) == 0:
            return nx.Graph(), points

        # 构建图
        G = nx.Graph()

        # 添加节点
        for i, point in enumerate(points):
            G.add_node(i, pos=tuple(point))

        # 使用 KDTree 查找邻近点（8-连通）
        tree = cKDTree(points)

        for i, point in enumerate(points):
            # 查找距离 <= sqrt(2) 的邻居（8-连通）
            neighbors = tree.query_ball_point(point, r=1.5)
            for j in neighbors:
                if i < j:  # 避免重复边
                    dist = np.linalg.norm(points[i] - points[j])
                    G.add_edge(i, j, weight=dist)

        return G, points

    def calculate_apls(self, gt_label, pred_label, sample_points=50):
        """
        计算 APLS (Average Path Length Similarity)

        参数:
            gt_label: 真实标签
            pred_label: 预测结果
            sample_points: 采样点对数量
        """
        # 骨架化
        gt_skel = skeletonize(gt_label > 0).astype(np.uint8)
        pred_skel = skeletonize(pred_label > 0).astype(np.uint8)

        # 转换为图
        gt_graph, gt_points = self.skeleton_to_graph(gt_skel)
        pred_graph, pred_points = self.skeleton_to_graph(pred_skel)

        # 如果任一图为空
        if len(gt_points) == 0 or len(pred_points) == 0:
            if len(gt_points) == 0 and len(pred_points) == 0:
                return 1.0  # 都为空，完美匹配
            else:
                return 0.0  # 一个为空，完全不匹配

        # 随机采样点对
        n_gt = len(gt_points)
        n_sample = min(sample_points, n_gt * (n_gt - 1) // 2)

        if n_sample == 0:
            return 1.0

        # 生成随机点对索引
        np.random.seed(42)  # 固定随机种子保证可重复性
        sampled_pairs = []

        if n_sample < n_gt * (n_gt - 1) // 2:
            # 随机采样
            while len(sampled_pairs) < n_sample:
                i, j = np.random.choice(n_gt, 2, replace=False)
                if i != j and (i, j) not in sampled_pairs and (j, i) not in sampled_pairs:
                    sampled_pairs.append((i, j))
        else:
            # 使用所有点对
            for i in range(n_gt):
                for j in range(i + 1, n_gt):
                    sampled_pairs.append((i, j))

        # 计算路径长度差异
        path_diffs = []

        for i, j in sampled_pairs:
            # 计算真实图中的最短路径
            try:
                gt_path_length = nx.shortest_path_length(gt_graph, i, j, weight='weight')
            except nx.NetworkXNoPath:
                gt_path_length = float('inf')

            # 在预测图中找到最近的对应点
            gt_point_i = gt_points[i]
            gt_point_j = gt_points[j]

            # 使用 KDTree 找到预测图中最近的点
            if len(pred_points) > 0:
                tree_pred = cKDTree(pred_points)
                _, pred_i = tree_pred.query(gt_point_i)
                _, pred_j = tree_pred.query(gt_point_j)

                # 计算预测图中的最短路径
                try:
                    pred_path_length = nx.shortest_path_length(pred_graph, pred_i, pred_j, weight='weight')
                except nx.NetworkXNoPath:
                    pred_path_length = float('inf')
            else:
                pred_path_length = float('inf')

            # 计算路径长度差异
            if gt_path_length == float('inf') and pred_path_length == float('inf'):
                diff = 0  # 都不连通，认为一致
            elif gt_path_length == float('inf') or pred_path_length == float('inf'):
                diff = 1  # 一个连通一个不连通，最大差异
            else:
                # 归一化差异
                diff = min(1, abs(gt_path_length - pred_path_length) / max(gt_path_length, pred_path_length))

            path_diffs.append(diff)

        # APLS = 1 - 平均路径差异
        apls = 1 - np.mean(path_diffs)
        return apls

    def calculate_connectivity(self, gt_label, pred_label):
        """计算连通性指标（仅对有前景的图）"""
        # 检查是否有前景
        if np.sum(gt_label) == 0:
            return  # 纯背景图，跳过连通性计算

        self.foreground_images += 1

        # 骨架化
        label_skel = skeletonize(gt_label > 0).astype(np.uint8)
        pred_skel = skeletonize(pred_label > 0).astype(np.uint8)

        # 标注连通组件（使用重命名的函数）
        label_components = sk_label(label_skel, connectivity=2)
        pred_components = sk_label(pred_skel, connectivity=2)

        n_label = label_components.max()
        n_pred = pred_components.max()

        # 如果标签有前景但预测为空
        if n_pred == 0:
            self.connectivity_scores.append(0.0)
            self.completeness_scores.append(0.0)
            return

        # 连通性：预测组件与标签组件的匹配度
        matched_pred = 0
        for i in range(1, n_pred + 1):
            pred_mask = (pred_components == i)
            overlap = np.sum(pred_mask & (label_skel > 0))
            if overlap > 0:
                matched_pred += 1

        connectivity = matched_pred / n_pred

        # 完整性：标签组件被预测覆盖的比例
        matched_label = 0
        for i in range(1, n_label + 1):
            label_mask = (label_components == i)
            overlap = np.sum(label_mask & (pred_skel > 0))
            if overlap > 0:
                matched_label += 1

        completeness = matched_label / n_label

        self.connectivity_scores.append(connectivity)
        self.completeness_scores.append(completeness)

    def evaluate(self):
        """评估整个测试集"""
        label_files = sorted([f for f in os.listdir(self.label_dir)
                              if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])

        if len(label_files) == 0:
            raise ValueError(f"在 {self.label_dir} 中未找到图像文件")

        print(f"找到 {len(label_files)} 张测试图像")

        for filename in tqdm(label_files, desc="评估中"):
            label_path = os.path.join(self.label_dir, filename)
            pred_path = os.path.join(self.pred_dir, filename[:-4]+'.tif')

            if not os.path.exists(pred_path):
                print(f"\n警告: 未找到预测结果 {pred_path}，跳过")
                continue

            try:
                gt_label = self.load_image(label_path)
                pred_label = self.load_image(pred_path)
            except Exception as e:
                print(f"\n警告: 读取 {filename} 失败: {e}，跳过")
                continue

            if gt_label.shape != pred_label.shape:
                print(f"\n警告: {filename} 尺寸不匹配 (label: {gt_label.shape}, pred: {pred_label.shape})，跳过")
                continue

            # 累计混淆矩阵
            self.accumulate_confusion_matrix(gt_label, pred_label)

            # 计算连通性（仅对有前景的图）
            self.calculate_connectivity(gt_label, pred_label)

            # 计算 APLS（仅对有前景的图）
            if np.sum(gt_label) > 0:
                try:
                    apls = self.calculate_apls(gt_label, pred_label)
                    self.apls_scores.append(apls)
                except Exception as e:
                    print(f"\n警告: 计算 APLS 失败 ({filename}): {e}")

            self.total_images += 1

        return self.get_results()

    def get_results(self):
        """基于全局累计的混淆矩阵计算最终指标"""
        if self.total_images == 0:
            raise ValueError("没有成功评估任何图像")

        results = {}

        # ===== 像素级指标（基于全局TP/FP/FN/TN） =====
        precision = self.TP / (self.TP + self.FP) if (self.TP + self.FP) > 0 else 0
        recall = self.TP / (self.TP + self.FN) if (self.TP + self.FN) > 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        # 前景IoU
        intersection_fg = self.TP
        union_fg = self.TP + self.FP + self.FN
        iou_fg = intersection_fg / union_fg if union_fg > 0 else 0

        # 背景IoU
        intersection_bg = self.TN
        union_bg = self.TN + self.FP + self.FN
        iou_bg = intersection_bg / union_bg if union_bg > 0 else 0

        # 平均IoU
        miou = (iou_fg + iou_bg) / 2

        results['pixel_metrics'] = {
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1_score,
            'Foreground IoU': iou_fg,
            'Background IoU': iou_bg,
            'Mean IoU (mIoU)': miou
        }

        # ===== 连通性指标（仅基于有前景的图） =====
        if len(self.connectivity_scores) > 0:
            avg_connectivity = np.mean(self.connectivity_scores)
            avg_completeness = np.mean(self.completeness_scores)
        else:
            avg_connectivity = 0.0
            avg_completeness = 0.0

        # APLS 指标
        if len(self.apls_scores) > 0:
            avg_apls = np.mean(self.apls_scores)
        else:
            avg_apls = 0.0

        results['connectivity_metrics'] = {
            'Connectivity': avg_connectivity,
            'Completeness': avg_completeness,
            'APLS': avg_apls,
            'Foreground Images': self.foreground_images,
            'Total Images': self.total_images
        }

        # ===== 混淆矩阵统计 =====
        results['confusion_matrix'] = {
            'TP': int(self.TP),
            'FP': int(self.FP),
            'FN': int(self.FN),
            'TN': int(self.TN)
        }

        return results

    def print_results(self, results):
        """格式化打印结果"""
        print("\n" + "=" * 60)
        print("管线提取精度评价结果")
        print("=" * 60)

        print("\n【像素级指标】（基于全局混淆矩阵）")
        for key, value in results['pixel_metrics'].items():
            print(f"  {key:20s}: {value:.4f} ({value * 100:.2f}%)")

        print("\n【连通性与拓扑指标】（仅统计有前景的图像）")
        conn = results['connectivity_metrics']
        print(f"  {'Connectivity':20s}: {conn['Connectivity']:.4f} ({conn['Connectivity'] * 100:.2f}%)")
        print(f"  {'Completeness':20s}: {conn['Completeness']:.4f} ({conn['Completeness'] * 100:.2f}%)")
        print(f"  {'APLS':20s}: {conn['APLS']:.4f} ({conn['APLS'] * 100:.2f}%)")
        print(f"  有前景图像数: {conn['Foreground Images']} / {conn['Total Images']}")

        print("\n【混淆矩阵统计】")
        cm = results['confusion_matrix']
        print(f"  TP (真阳性): {cm['TP']:,}")
        print(f"  FP (假阳性): {cm['FP']:,}")
        print(f"  FN (假阴性): {cm['FN']:,}")
        print(f"  TN (真阴性): {cm['TN']:,}")

        print("=" * 60 + "\n")


# ===== 使用示例 =====
if __name__ == "__main__":
    evaluator = PipelineEvaluator(
        label_dir=r"E:\PipelineRS\labels",
        pred_dir=r"your prediction masks folder"
    )

    results = evaluator.evaluate()
    evaluator.print_results(results)

    # 可选：保存结果到JSON
    import json

    with open("evaluation_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    print("结果已保存到 evaluation_results.json")
