# TED视频自动分割系统

## 项目概述

本项目是一个基于计算机视觉和深度学习的视频自动分割系统，专门用于从TED演讲视频中提取高质量的单人演讲片段。系统集成了**智能爬虫**、**六阶段筛选流程**和**安全处理**功能，采用YOLO目标检测、MediaPipe姿态估计、DWPose多人检测和场景切换检测技术，实现了从视频爬取到高质量片段输出的完整端到端解决方案。

## 💡 核心特性

### 🔄 完整工作流程
- **智能爬虫**: 自动爬取TED频道最新视频，避免重复处理
- **自动下载**: 智能下载高质量视频文件
- **六阶段筛选**: 多层次质量控制，确保输出片段高质量
- **安全处理**: 自动备份、重复检测、增量处理
- **格式输出**: 生成EchoMimicV2兼容的时间码文件

### 🧠 核心算法

1. **YOLO v8** - 人体目标检测，快速识别包含人物的帧
2. **MediaPipe** - 实时姿态估计，检测上半身关键点
3. **DWPose ONNX** - 多人检测和姿态分析，排除多人场景
4. **场景切换检测** - 基于直方图和边缘特征的场景连续性分析

### 🎯 六阶段筛选流程

1. **Stage 1: YOLO人体检测** - 初步筛选包含人物的帧
2. **Stage 2: 几何质量评估** - 分析人体位置、大小、面部清晰度等几何特征
3. **Stage 3: MediaPipe姿态估计** - 检测上半身关键点，确保姿态完整性
4. **Stage 4: 姿态质量评估** - 综合姿态特征进行质量评分
5. **Stage 5: DWPose多人检测** - 排除多人场景，确保单人演讲环境
6. **Stage 6: 场景切换检测** - 检测并过滤包含镜头切换的片段

## 项目结构

```
ted_video_segmentation/
├── main.py                           # 主程序入口
├── video_segmentation_pipeline.py    # 视频处理管道
├── configs/
│   └── segmentation_parameters.yaml  # 配置参数
├── methods/
│   ├── yolo_mediapipe.py             # 核心检测算法
│   ├── dwpose_onnx_detector.py       # DWPose检测器
│   └── scene_change_detector.py      # 场景切换检测
├── models/
│   └── dwpose/                       # DWPose模型配置
├── utils/                            # 工具函数
├── ffmpeg/                           # 视频处理工具
├── flowcharts/                       # 算法流程图
└── mermaid_codes/                    # 流程图源码
```

## 环境要求

### 系统要求
- Python 3.8+
- Windows/Linux/macOS
- GPU支持（推荐，用于加速推理）

### 依赖库
```bash
pip install -r requirements.txt
```

主要依赖：
- torch >= 1.13.0
- ultralytics (YOLO v8)
- mediapipe >= 0.10.0
- onnxruntime-gpu
- opencv-python
- numpy
- pyyaml
- yt-dlp

## 🚀 快速开始

### 环境安装

```bash
# 克隆项目
git clone https://github.com/15982002029/ted-video-segmentation.git
cd ted-video-segmentation

# 安装依赖
pip install -r requirements.txt
```

### 必需文件下载

**重要：** 由于文件体积较大，以下文件需要手动下载：

#### 1. FFmpeg
- 下载地址：https://ffmpeg.org/download.html
- 解压后将 `ffmpeg.exe` 和 `ffprobe.exe` 放到 `ffmpeg/` 目录
- 或者使用包管理器：`conda install ffmpeg` 

#### 2. DWPose ONNX模型文件
```bash
# 创建模型目录（如果不存在）
mkdir -p models/dwpose

# 下载模型文件（约100MB）
wget -O models/dwpose/dw-ll_ucoco_384.onnx "https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx"
wget -O models/dwpose/yolox_l.onnx "https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx"
```

**Windows用户可以手动下载：**
- [dw-ll_ucoco_384.onnx](https://huggingface.co/yzd-v/DWPose/resolve/main/dw-ll_ucoco_384.onnx)
- [yolox_l.onnx](https://huggingface.co/yzd-v/DWPose/resolve/main/yolox_l.onnx)

**注意：** YOLOv8模型会在首次运行时自动下载。

### 基本使用

```bash
# 🎯 交互式运行（推荐新手）
python main.py

# 🔄 完整工作流程（爬虫 + 处理）
python main.py --workflow --max-videos 10

# 📂 处理指定视频文件
python main.py --input video_urls.txt --max-videos 5

# 🆕 仅爬虫模式
python main.py --crawl --crawl-max 20
```

### 高级用法

```bash
# 处理本地视频文件（downloads文件夹）
python main.py --local-videos --max-videos 10

# 禁用安全模式（不推荐）
python main.py --workflow --disable-safety

# 自定义配置文件
python main.py --config configs/custom_parameters.yaml
```

### 输入格式

视频URL文件格式（每行一个YouTube链接）：
```
https://www.youtube.com/watch?v=video_id_1
https://www.youtube.com/watch?v=video_id_2
https://youtu.be/video_id_3
```

### 输出格式

**EchoMimicV2格式时间段文件**：
```csv
URL,Start Timecode,End Timecode
https://www.youtube.com/watch?v=video_id,00:02:28.750,00:02:45.625
https://www.youtube.com/watch?v=video_id,00:05:12.250,00:05:28.875
```

**生成的文件**：
- `filtered_ted_segments.txt` - 主输出文件
- `results/reports/video_segmentation_report.md` - 详细处理报告
- `results/video_segments/` - 切割的视频片段文件

## 算法参数

系统提供丰富的参数配置，主要包括：

### YOLO检测参数
- `yolo_confidence_threshold`: 0.5 - 人体检测置信度阈值

### 几何质量参数  
- `min_person_area_ratio`: 0.15 - 最小人体面积比例
- `min_person_height_ratio`: 0.3 - 最小人体高度比例
- `max_face_aspect_ratio`: 5 - 最大面部宽高比

### MediaPipe参数
- `mediapipe_detection_confidence`: 0.7 - 姿态检测置信度
- `stage3_visibility_threshold`: 0.7 - 关键点可见度阈值

### DWPose参数
- `person_confidence_threshold`: 0.3 - 人体检测阈值
- `keypoint_confidence_threshold`: 0.3 - 关键点置信度阈值

## 📊 性能指标

### 六阶段通过率统计

根据大规模测试结果，系统在不同阶段的通过率如下：

| 阶段 | 功能 | 通过率 | 说明 |
|-----|------|--------|------|
| **Stage 1** | YOLO人体检测 | ~19% | 初步筛选包含人物的帧 |
| **Stage 2** | 几何质量评估 | ~57% | 人体位置、大小、面部清晰度评估 |
| **Stage 3** | MediaPipe姿态估计 | ~68% | 上半身关键点检测 |
| **Stage 4** | 姿态质量评估 | ~97% | 综合姿态特征质量评分 |
| **Stage 5** | DWPose多人检测 | ~99% | 排除多人场景 |
| **Stage 6** | 场景切换检测 | ~100% | 过滤镜头切换片段 |

**最终综合通过率**: 约8-10%，确保了输出片段的高质量

### 典型处理结果

**处理10个TED视频的性能示例**：
- 📹 处理视频总数：10个
- ✅ 生成有效片段：23个  
- 🎯 片段质量分数：0.929-0.964
- ⏱️ 总片段时长：约8.2分钟
- 🚀 平均处理速度：约1-2分钟/视频（取决于硬件配置）

## 🔧 技术特点

### 核心优势
1. **🔄 端到端自动化**: 从视频爬取到片段输出的完整自动化流程
2. **🧠 多模态融合**: 结合目标检测、姿态估计和场景分析
3. **🛡️ 鲁棒性设计**: 六阶段筛选确保结果可靠性
4. **⚙️ 参数可配置**: 支持针对不同场景的参数调优
5. **✨ 高质量输出**: 严格的质量控制机制
6. **🚀 高效处理**: 优化的算法流程支持快速处理
7. **🛡️ 安全机制**: 自动备份、重复检测、增量处理

### 适用场景
- 🎓 学术研究中的视频数据准备
- 🤖 AI训练数据集构建
- 📺 视频内容自动化处理
- 🎬 多媒体内容分析

## 📋 项目结构

```
ted_video_segmentation/
├── main.py                              # 🚀 主程序入口
├── video_segmentation_pipeline.py       # ⚙️ 核心处理管道
├── links.py                             # 🕷️ 智能爬虫模块
├── configs/
│   └── segmentation_parameters.yaml     # 📄 系统配置文件
├── methods/                             # 🧠 核心算法模块
│   ├── yolo_mediapipe.py                # 🎯 YOLO+MediaPipe检测
│   ├── dwpose_onnx_detector.py          # 👥 DWPose多人检测
│   └── scene_change_detector.py         # 🎬 场景切换检测
├── utils/                               # 🛠️ 工具函数库
│   ├── smart_downloader.py              # 📥 智能视频下载器
│   ├── video_utils.py                   # 🎥 视频处理工具
│   └── result_processor.py              # 📊 结果处理器
├── models/dwpose/                       # 🤖 DWPose模型配置
├── ffmpeg/                              # 🎬 FFmpeg工具
├── flowcharts/                          # 📊 算法流程图
└── mermaid_codes/                       # 📈 流程图源码
```

## 📈 算法流程图

项目包含详细的算法流程图文档：
- 📊 `flowcharts/02_safe_processing_stage.png` - 安全处理阶段流程
- 📊 `flowcharts_en/` - 英文版流程图
- 📝 `mermaid_codes/` - 流程图源码

## 🔗 相关资源

### 模型和依赖
- [YOLOv8](https://github.com/ultralytics/ultralytics) - 人体目标检测
- [MediaPipe](https://mediapipe.dev/) - 实时姿态估计
- [DWPose](https://github.com/IDEA-Research/DWPose) - 多人姿态检测
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) - YouTube视频下载

## 📜 引用

如果您在研究中使用了本项目，请引用：

```bibtex
@software{ted_video_segmentation,
  title={TED Video Automatic Segmentation System},
  subtitle={A Multi-stage Computer Vision Pipeline for High-quality Speaker Segment Extraction},
  year={2024},
  url={https://github.com/your-username/ted-video-segmentation}
}
```

## 📧 联系方式

- 📝 技术问题或建议：请通过 [GitHub Issues](https://github.com/your-username/ted-video-segmentation/issues) 联系
- 📊 学术合作：欢迎通过邮件联系

## ⚠️ 免责声明

**注意**: 本项目仅用于学术研究目的，请遵守相关视频平台的使用条款和版权法律。使用本项目时请确保：
- 遵守YouTube服务条款
- 尊重视频内容的版权
- 仅将处理结果用于学术研究