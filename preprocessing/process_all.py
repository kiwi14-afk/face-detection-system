import cv2
import numpy as np
import os
import glob


class FaceImagePreprocessor:
    def __init__(self,
                 clip_limit=3.0, tile_grid_size=(8, 8),
                 bf_d=5, bf_sigma_color=25, bf_sigma_space=25,
                 usm_amount=1.5, usm_radius=1.5):
        """
        初始化预处理器，已配置为最优基准线参数，在去噪与保留高频特征之间取得平衡。
        """
        # 初始化 CLAHE (光照补偿)
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

        # 双边滤波参数 (去噪)
        self.bf_d = bf_d
        self.bf_sigma_color = bf_sigma_color
        self.bf_sigma_space = bf_sigma_space

        # USM 锐化参数 (边缘增强)
        self.usm_amount = usm_amount
        self.usm_radius = usm_radius

    def process_image(self, image):
        # 转换到 LAB 颜色空间 (仅对亮度通道 L 进行处理，避免色彩失真)
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # 光照补偿 (CLAHE) - 增强阴影区域细节
        l_clahe = self.clahe.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        img_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        # 双边滤波去噪 - 降低滤波强度，保护人脸五官边缘
        img_denoised = cv2.bilateralFilter(
            img_clahe, self.bf_d, self.bf_sigma_color, self.bf_sigma_space
        )
        # USM 锐化 - 针对保留下来的细节进行适度增强
        gaussian_blur = cv2.GaussianBlur(img_denoised, (0, 0), self.usm_radius)
        img_sharpened = cv2.addWeighted(
            img_denoised, self.usm_amount,
            gaussian_blur, 1.0 - self.usm_amount, 0
        )
        return img_sharpened


def batch_process_full_dataset(input_dir, output_dir):
    """
    全量遍历并处理完整数据集，保留原始子目录结构
    """
    # 递归查找所有 jpg 图片
    search_pattern = os.path.join(input_dir, '**', '*.jpg')
    image_paths = glob.glob(search_pattern, recursive=True)

    total_images = len(image_paths)
    if total_images == 0:
        print("未找到任何图片，请检查 input_dir 路径是否正确。")
        return

    print(f"开始全量处理，共发现 {total_images} 张图片...")

    preprocessor = FaceImagePreprocessor()
    success_count = 0

    for i, path in enumerate(image_paths):
        img = cv2.imread(path)
        if img is None:
            print(f"警告：无法读取图像 {path}")
            continue

        # 执行预处理算法
        processed_img = preprocessor.process_image(img)

        # 获取相对路径以保持原有的目录结构
        relative_path = os.path.relpath(path, input_dir)
        output_path = os.path.join(output_dir, relative_path)

        # 确保输出子文件夹存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, processed_img)
        success_count += 1

        # 每处理 500 张打印一次进度
        if (i + 1) % 500 == 0 or (i + 1) == total_images:
            percent = (i + 1) / total_images * 100
            print(f"正在处理: {i + 1}/{total_images} ({percent:.1f}%)")

    print(f" 全量数据批处理完成！成功处理并保存了 {success_count} 张图像。")
    print(f"请前往 {output_dir} 查看，这是最终交付给模型训练的完整数据集。")


if __name__ == "__main__":
    # 存放原图的根目录
    INPUT_FOLDER = "../data/raw_images"

    # 最终完整版预处理数据集的输出目录
    OUTPUT_FOLDER = "../data/processed_images"

    batch_process_full_dataset(INPUT_FOLDER, OUTPUT_FOLDER)