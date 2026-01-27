# 多模态情感分类项目

## 项目简介

本项目基于CLIP（Contrastive Language-Image Pretraining）模型实现图像-文本多模态情感分类，支持对社交媒体内容进行积极（positive）、消极（negative）和中性（neutral）的情感分析。

主要功能包括：
- 多种融合策略的模型训练（latefusion、cross_attention、clipfusion）
- 超参数调优（学习率、批次大小、文本编码器冻结状态）
- 基线模型训练（仅图像、仅文本）
- 测试集预测与结果生成

## 目录结构

```
code/              # 代码和数据目录
├── data/          # 数据目录（包含图像和文本文件）
│   ├── *.jpg      # 图像文件（以guid命名）
│   └── *.txt      # 文本文件（以guid命名）
├── cleaned/       # 清理后的数据
│   ├── train_clean.txt # 清理后的训练数据
│   └── val_clean.txt   # 清理后的验证数据
├── results/       # 训练结果目录
│   ├── latefusion/    # latefusion模型训练结果
│   ├── cross_attention/ # cross_attention模型训练结果
│   └── clipfusion/     # clipfusion模型训练结果
├── predictions/   # 预测结果目录
├── predict_test.py    # 测试集预测脚本
├── train_latefusion.py # latefusion模型训练脚本
├── train_cross_attention.py # cross_attention模型训练脚本
├── train_clipfusion_full.py # clipfusion模型训练脚本
├── train.txt         # 原始训练数据
└── test_without_label.txt # 测试文件（包含null标签）
```

## 依赖项

使用以下命令安装项目依赖：

```bash
pip install -r requirements.txt
```

主要依赖包括：
- PyTorch 2.0.0+
- Transformers 4.30.0+
- Pandas 2.0.0+
- NumPy 1.24.0+
- Pillow 9.0.0+
- tqdm 4.65.0+

## 使用说明

所有命令需要在 `code` 目录下执行：

```bash
cd code
```

## 数据准备

### 数据格式

1. **原始训练数据** (`train.txt`)：
```csv
guid,tag
4597,negative
26,neutral
```

2. **清理后的训练/验证数据** (`cleaned/train_clean.txt`, `cleaned/val_clean.txt`)：
```csv
guid	text_clean	tag	img_path	img_exists	img_ok
1801		positive	data/1801.jpg	True	True
```

3. **测试数据** (`test_without_label.txt`)：
```csv
guid,null
8,null
1576,null
```

4. **图像和文本文件**：
   - 图像文件：`data/{guid}.jpg`
   - 文本文件：`data/{guid}.txt`

### 数据清理

使用 `data_clean_split.py` 脚本对原始数据进行清理和分割：

```bash
python data_clean_split.py
```

## 模型训练

本项目支持多种模型架构，包括：

1. **latefusion**：简单的特征拼接融合策略
2. **cross_attention**：基于交叉注意力的融合策略
3. **clipfusion**：基于CLIP特征的融合策略
4. **baselines**：仅图像或仅文本的基线模型

### latefusion模型训练

使用 `train_latefusion.py` 脚本进行训练：

```bash
# 示例1：学习率1e-4，批次大小16，不冻结文本编码器
python train_latefusion.py --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --epochs 10 --batch-size 16 --lr 1e-4 --output-dir results/latefusion/lr1e-4_bs16_nofreeze --output-path best_latefusion_lr1e-4_bs16_nofreeze.pt

# 示例2：学习率5e-5，批次大小32，冻结文本编码器
python train_latefusion.py --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --epochs 10 --batch-size 32 --lr 5e-5 --freeze-text --output-dir results/latefusion/lr5e-5_bs32_freeze --output-path best_latefusion_lr5e-5_bs32_freeze.pt
```

### cross_attention模型训练

使用 `train_cross_attention.py` 脚本进行训练：

```bash
python train_cross_attention.py --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --epochs 10 --batch-size 4 --lr 1e-5 --output-dir results/cross_attention/lr1e-5_bs4_freeze --output best_cross_attention_lr1e-5_bs4_freeze.pt
```

### clipfusion模型训练

使用 `train_clipfusion_full.py` 脚本进行训练：

```bash
# 示例1：学习率1e-4，批次大小32，冻结CLIP模型
python train_clipfusion_full.py --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --epochs 10 --batch-size 32 --lr 1e-4 --freeze-clip --output-dir results/clipfusion/lr1e-4_bs32_freeze --output best_clipfusion_lr1e-4_bs32_freeze.pt

# 示例2：学习率1e-5，批次大小16，不冻结CLIP模型
python train_clipfusion_full.py --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --epochs 10 --batch-size 16 --lr 1e-5 --output-dir results/clipfusion/lr1e-5_bs16_nofreeze --output best_clipfusion_lr1e-5_bs16_nofreeze.pt
```

### 基线模型训练

使用 `train_baselines.py` 脚本进行训练：

```bash
# 仅文本模型
python train_baselines.py --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --mode text --epochs 10 --batch-size 32 --lr 1e-4 --output-dir results/baselines/text_only --output-path best_text_only.pt

# 仅图像模型
python train_baselines.py --train-file cleaned/train_clean.txt --val-file cleaned/val_clean.txt --data-dir data --mode image --epochs 10 --batch-size 32 --lr 1e-4 --output-dir results/baselines/image_only --output-path best_image_only.pt
```

## 模型预测

使用 `predict_test.py` 脚本生成测试集预测结果：

```bash
python predict_test.py --test-file test_without_label.txt --data-dir data --clip-model openai/clip-vit-base-patch32 --checkpoint best_clipfusion_lr1e-4_bs32_freeze.pt --batch-size 32 --output-dir predictions
```

预测结果将保存到 `predictions/test_predictions.txt` 文件中，格式如下：

```csv
guid,label
8,positive
1576,negative
2320,neutral
```

## 超参数调优

本项目支持多种超参数组合的训练，建议测试以下组合：

### latefusion模型

| 学习率 | 批次大小 | 冻结文本编码器 |
|--------|----------|----------------|
| 1e-4 | 16 | 否 |
| 1e-4 | 32 | 否 |
| 5e-5 | 16 | 否 |
| 5e-5 | 32 | 否 |
| 1e-4 | 16 | 是 |
| 1e-4 | 32 | 是 |
| 5e-5 | 16 | 是 |
| 5e-5 | 32 | 是 |

### cross_attention模型

| 学习率 | 批次大小 |
|--------|----------|
| 1e-5 | 4 |
| 2e-5 | 4 |
| 1e-5 | 8 |

### clipfusion模型

| 学习率 | 批次大小 | 冻结CLIP模型 |
|--------|----------|--------------|
| 1e-4 | 32 | 是 |
| 1e-4 | 16 | 是 |
| 5e-5 | 32 | 是 |
| 1e-5 | 16 | 否 |
| 1e-5 | 8 | 否 |

## 硬件要求

- 建议使用GPU进行训练，至少8GB显存
- 训练时间：每个模型约10-30分钟（取决于超参数设置）

## 故障排除

### 常见问题

1. **文本文件编码错误**：
   - 症状：训练时出现编码错误
   - 解决方案：确保文本文件使用UTF-8编码

2. **图像文件不存在**：
   - 症状：训练时出现文件不存在错误
   - 解决方案：确保data目录下存在对应的图像文件

3. **预测结果全部为中性**：
   - 症状：所有预测结果都为"neutral"
   - 解决方案：检查文本输入是否为空，确保每个样本都有有效的文本输入

### 调试技巧

- 使用 `eda_quick.py` 脚本快速探索数据分布
- 调整批次大小以适应不同的硬件配置

## 结果评估

训练完成后，可以通过以下方式评估模型性能：

1. **验证集准确率**：训练过程中会计算验证集的准确率
2. **混淆矩阵**：分析模型在不同类别上的表现
3. **测试集预测**：生成测试集的预测结果并保存

## 可视化

使用 `train_clipfusion_full.py` 脚本的可视化功能可以生成错误分析的可视化结果：

```bash
# 生成可视化结果
python train_clipfusion_full.py --visualize --error-analysis-file results/clipfusion/lr1e-4_bs32_freeze/error_analysis.json --output best_clipfusion_lr1e-4_bs32_freeze.pt --clip-model openai/clip-vit-base-patch32 --output-dir results/clipfusion/lr1e-4_bs32_freeze --data-dir data
```

### 可视化结果说明

- **输出目录**：`results/clipfusion/lr1e-4_bs32_freeze/visualizations/`
- **生成文件**：每个错误样本的可视化图片，包含图像、文本、真实标签和预测标签
- **可视化内容**：
  - 原始图像
  - 模型预测结果
  - 真实标签
  - 模型输出的logits和概率
