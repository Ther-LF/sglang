# SVG2 Multi-Config Top-p Testing Guide

本指南说明如何使用更新后的脚本测试多个 top-p 配置。

## 📋 快速开始

### 基本用法（测试默认的4个top-p值）

```bash
cd /path/to/sglang/examples/multimodal_gen/scripts
./svg2_wan_i2v_720p.sh
```

**默认配置**：测试 top-p = 0.3, 0.5, 0.7, 0.9

## 🎛️ 自定义选项

### 1. 自定义 top-p 值

```bash
# 只测试 3 个值
TOP_P_VALUES="0.5,0.7,0.9" ./svg2_wan_i2v_720p.sh

# 测试更激进的配置
TOP_P_VALUES="0.3,0.5" ./svg2_wan_i2v_720p.sh

# 测试单个值
TOP_P_VALUES="0.7" ./svg2_wan_i2v_720p.sh
```

### 2. 使用示例提示词

```bash
# 使用 Sparse-VideoGen 的示例 1
PROMPT_ID=1 ./svg2_wan_i2v_720p.sh

# 使用示例 2
PROMPT_ID=2 ./svg2_wan_i2v_720p.sh
```

### 3. 自定义提示词和图片

```bash
PROMPT="A majestic eagle soaring through clouds" \
IMAGE_PATH="/path/to/your/image.jpg" \
./svg2_wan_i2v_720p.sh
```

### 4. 添加 Dense Baseline 对比

```bash
COMPARE_WITH_DENSE=true ./svg2_wan_i2v_720p.sh
```

### 5. 组合使用

```bash
TOP_P_VALUES="0.5,0.7,0.9" \
COMPARE_WITH_DENSE=true \
PROMPT_ID=3 \
NUM_GPUS=2 \
./svg2_wan_i2v_720p.sh
```

## 📊 输出结果

### 文件结构

脚本会生成以下文件：

```
outputs/svg2/wan_i2v/
└── Step_40-Res_720p/
    └── TFP_0.35-LFP_0.03/
        └── QC_300-KC_1000-TopP_Multi/
            └── Init_50-Step_2-MinR_0.10/
                ├── 1-0_p03.mp4  (top-p=0.3)
                ├── 1-0_p05.mp4  (top-p=0.5)
                ├── 1-0_p07.mp4  (top-p=0.7)
                ├── 1-0_p09.mp4  (top-p=0.9)
                └── 1-0_dense.mp4 (如果启用了 COMPARE_WITH_DENSE)
```

### 性能报告

运行结束后会显示对比报告：

```
================================================================================
 Performance Comparison Summary
================================================================================
 Configuration                  | Time (s)     | Speedup vs Dense  
--------------------------------------------------------------------------------
 Dense (FlashAttn2)             | 45.32        | Baseline (1.00x)  
 SVG2 (p=0.3)                   | 10.23        | 4.43x             
 SVG2 (p=0.5)                   | 12.45        | 3.64x             
 SVG2 (p=0.7)                   | 15.23        | 2.98x             
 SVG2 (p=0.9)                   | 18.67        | 2.43x             
================================================================================

 SVG2 Top-p Trade-off Analysis:
 ------------------------------------------------------------------------------
   Fastest SVG2 Config: SVG2 (p=0.3) (10.23s)
   Slowest SVG2 Config: SVG2 (p=0.9) (18.67s)
   Performance Variance: 82.5%
 ------------------------------------------------------------------------------
```

## 🔧 高级配置

### 环境变量完整列表

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `TOP_P_VALUES` | `"0.3,0.5,0.7,0.9"` | 要测试的 top-p 值（逗号分隔） |
| `COMPARE_WITH_DENSE` | `false` | 是否包含 Dense baseline |
| `PROMPT_ID` | `1` | 使用的示例 ID |
| `PROMPT` | 默认提示词 | 自定义提示词 |
| `IMAGE_PATH` | 基于 PROMPT_ID | 输入图片路径 |
| `NUM_GPUS` | `1` | GPU 数量 |
| `SVG_BASE` | `"/root/Sparse-VideoGen"` | Sparse-VideoGen 路径 |
| `MODEL_ID` | Wan2.1 模型路径 | 模型路径 |

### 修改 SVG2 参数

编辑脚本中的配置段：

```bash
# K-Means clustering parameters
qc_kmeans=300      # Number of query clusters
kc_kmeans=1000     # Number of key clusters
min_kc_ratio=0.10  # Minimum ratio of key clusters to keep

# K-Means iteration settings
kmeans_iter_init=50
kmeans_iter_step=2

# Dense attention warm-up
first_times_fp=0.35
first_layers_fp=0.03
```

## 💡 使用建议

### 选择 top-p 值的指南

- **0.3-0.5**: 最快速度，适合预览和快速迭代
- **0.7**: 速度和质量的平衡点
- **0.9**: 接近原始质量，仍有显著加速

### 推荐测试流程

1. **初始测试**：使用默认配置测试所有 top-p 值
   ```bash
   COMPARE_WITH_DENSE=true ./svg2_wan_i2v_720p.sh
   ```

2. **质量评估**：比较生成的视频，选择质量可接受的最快配置

3. **批量生成**：使用选定的 top-p 值进行批量生成
   ```bash
   TOP_P_VALUES="0.7" ./svg2_wan_i2v_720p.sh
   ```

## 🐛 故障排除

### 常见问题

1. **图片未找到**
   ```bash
   # 检查 SVG_BASE 设置
   SVG_BASE="/path/to/Sparse-VideoGen" ./svg2_wan_i2v_720p.sh
   ```

2. **OOM 错误**
   - 减少测试的 top-p 值数量
   - 使用更小的分辨率（修改脚本中的 `resolution="480p"`）

3. **模型未找到**
   ```bash
   MODEL_ID="/path/to/your/model" ./svg2_wan_i2v_720p.sh
   ```

## 📝 示例工作流

```bash
# 1. 快速测试单个 prompt
PROMPT_ID=1 TOP_P_VALUES="0.7" ./svg2_wan_i2v_720p.sh

# 2. 全面对比测试
PROMPT_ID=1 COMPARE_WITH_DENSE=true ./svg2_wan_i2v_720p.sh

# 3. 批量测试多个 prompts
for id in 1 2 3 4 5; do
    PROMPT_ID=$id TOP_P_VALUES="0.7,0.9" ./svg2_wan_i2v_720p.sh
done
```

---

**注意**：首次运行会下载模型和编译 kernels，可能需要较长时间。

