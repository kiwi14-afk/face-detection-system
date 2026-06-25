# Face Detection Lab

基于 YuNet 的轻量级人脸检测系统，包含图像预处理管线、PyTorch 训练、WIDER FACE 评测和 Web 交互应用。

**85K 参数 · Anchor-Free · 多尺度检测 · 实时推理**

---

## 功能特性

- **图像预处理管线**：CLAHE 光照补偿 → 双边滤波去噪 → USM 锐化增强
- **YuNet 人脸检测**：纯 PyTorch 实现，多尺度检测（stride 8/16/32），anchor-free 设计
- **Web 交互应用**：图片上传、预处理对比、置信度可视化分析
- **WIDER FACE 评测**：多阈值 Precision / Recall / F1 批量评测

## 项目结构

```
face-detection-system/
├── preprocessing/
│   └── process_all.py              # 预处理管线（CLAHE + 双边滤波 + USM）
├── training/
│   ├── yunet_torch/
│   │   ├── model.py                # YuNet 网络架构
│   │   ├── loss.py                 # 多任务损失函数（分类 + 回归 + Objectness）
│   │   ├── dataset.py              # WIDER FACE 数据加载器
│   │   └── train.py                # 训练循环 + ONNX 导出
│   ├── launch_full.py              # 训练启动脚本
│   ├── eval_new.py                 # WIDER FACE 评测脚本
│   ├── outputs/model_full/weights/
│   │   └── final_model.pth         # 训练好的模型权重
│   └── data/
│       ├── splits_full/            # 训练/验证集划分
│       └── wider_face_split/       # WIDER FACE 原始标注
└── webapp/
    ├── server.py                   # FastAPI 后端
    └── static/
        └── index.html              # 前端界面
```

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 1.10+
- OpenCV 4.5+
- FastAPI + uvicorn

### 安装依赖

```bash
pip install torch torchvision opencv-python fastapi uvicorn python-multipart pillow numpy tqdm
```

### 启动 Web 应用

```bash
cd webapp
python server.py
```

浏览器访问 `http://127.0.0.1:7860`，上传图片即可体验。

### 运行评测

```bash
cd training
python eval_new.py
```

在 WIDER FACE 验证集上输出多阈值 Precision / Recall / F1。

### 训练模型

```bash
cd training
python launch_full.py
```

使用预处理后的 WIDER FACE 数据集从头训练（100 epoch, batch=16, lr=0.01）。

## 预处理管线

| 步骤 | 算法 | 参数 | 作用 |
|------|------|------|------|
| 1 | CLAHE | clip_limit=3.0, tile=(8,8) | 光照补偿，增强阴影细节 |
| 2 | 双边滤波 | d=5, σ_color=25, σ_space=25 | 保边去噪 |
| 3 | USM 锐化 | amount=1.5, radius=1.5 | 边缘增强 |

在 LAB 色彩空间的 L 通道上做 CLAHE，避免色彩失真。三级串联在去噪与保留高频特征之间取得平衡。

## Web 界面

```
┌──────────────────────┐  ┌──────────────────────────────┐
│   上传图片（拖拽）    │  │  原始图          预处理后    │
│                      │  │  ┌─────────┐    ┌─────────┐  │
│  预处理 [开/关]      │  │  │         │    │         │  │
│  置信度阈值 ───●───  │  │  │ Canvas  │    │ Canvas  │  │
│                      │  │  └─────────┘    └─────────┘  │
│  [检测]              │  │                              │
│                      │  │  人脸数: 3/4   耗时: 45ms    │
│                      │  │                              │
│                      │  │  置信度对比                   │
│                      │  │  #1  0.92 → 0.98  ↑ +0.06   │
│                      │  │  #2  0.45 → 0.67  ↑ +0.22   │
│                      │  │  #3  0.23 → —     ✕ 消失    │
│                      │  │  #4  —    → 0.55  ＋ 新增   │
└──────────────────────┘  └──────────────────────────────┘
```

- 检测框颜色编码：**青色** ≥0.8、**黄色** ≥0.6、**红色** <0.6
- 置信度变化状态：↑ 提升 · ↓ 下降 · ≈ 不变 · ✕ 消失 · ＋ 新增

## API

### POST /api/detect

上传图片进行人脸检测。

**请求参数**（multipart/form-data）：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image` | file | 必填 | JPG/PNG 图片 |
| `preprocess` | bool | true | 是否启用预处理对比 |
| `threshold` | float | 0.10 | 置信度阈值 (0.01-0.99) |

**响应示例**：

```json
{
  "preprocess": true,
  "threshold": 0.10,
  "detections_orig": [
    { "bbox": [120, 80, 200, 240], "conf": 0.9234 }
  ],
  "detections_proc": [
    { "bbox": [120, 80, 200, 240], "conf": 0.9851 }
  ],
  "conf_comparison": [
    {
      "bbox": [120, 80, 200, 240],
      "orig_conf": 0.9234,
      "proc_conf": 0.9851,
      "diff": 0.0617,
      "status": "up"
    }
  ],
  "proc_image": "data:image/jpeg;base64,...",
  "time_orig_ms": 45.2,
  "time_proc_ms": 48.7,
  "time_total_ms": 120.5
}
```

## 模型信息

| 属性 | 值 |
|------|-----|
| 架构 | YuNet |
| 参数量 | ~85K |
| 输入尺寸 | 640×640 |
| 检测尺度 | stride 8 / 16 / 32 |
| 设计类型 | Anchor-Free |
| 输出 | 分类 + BBox + Objectness（无关键点） |
| 训练集 | 10,304 张（WIDER FACE 预处理） |
| 训练轮次 | 100 epoch |
| 置信度计算 | cls × objectness（sigmoid 后） |
| NMS | IoU 0.45 |
| 权重文件 | `training/outputs/model_full/weights/final_model.pth` |

## 分工

| 成员 | 模块 | 核心产出 |
|------|------|----------|
| 成员A | 数据准备 + 图像预处理 | `process_all.py`、WIDER FACE 数据集 |
| 成员B | 模型训练系统 | `model.py`、`loss.py`、`dataset.py`、`train.py`、模型权重 |
| 成员C | Web 应用 + 模型评测 | `server.py`、`index.html`、`eval_new.py` |

详见 [汇报指南.md](汇报指南.md)

## 许可证

本项目仅用于学术研究与课程作业。
