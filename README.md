# 360° Panoramic Video Adaptive Transmission

基于光流 + 视口代理 + 带宽自适应的全景视频瓦片级差分编码方案，面向机器人巡检场景。

## 核心思想

在机器人移动全景巡检中，将三个信号统一纳入瓦片（tile）级码率分配决策：

| 信号 | 作用 | 更新频率 |
|---|---|---|
| **光流** (Optical Flow) | 回答"哪些区域人眼追不上" | 帧级 |
| **视口代理** (Viewport Proxy) | 回答"人现在在看哪里" | 帧级 |
| **带宽估计** (Bandwidth Est.) | 回答"总共能投多少码率" | 秒级 |

**三者关系**：带宽决定总量，光流 + 视口决定分配。

光流大的区域（运动快）→ 人眼动态视觉敏锐度下降 → 可以降质省码率 → 把比特集中到重要区域（静止物体 = 巡检目标）。

## 系统架构

```
全景视频帧 → 光流计算 ──→ Tile 优先级融合 → QP 偏移 → Tile 差分编码 → 推流
    │                        ↑
    ├── 视口代理(Gaussian) ──┤
    └── 带宽估计(EWMA) → QP_base 自适应 ──┘
```

## 离线验证管线

```
scripts/
├── step1_optical_flow.py    # 光流计算 + 球面校正 + 瓦片划分
├── step2_qp_table.py        # QP 偏移表生成（融合 + 带宽模拟）
├── step3_jpeg_tile.py       # 瓦片级 JPEG 压缩模拟差分编码
└── step4_visualize.py       # 可视化（热力图 + 带宽曲线 + PSNR）
```

### 数据流

```
step1 → flow_data.json
step2 → qp_table.json
step3 → uniform_jpeg_baseline.mp4 + flow_guided_jpeg.mp4 + encode_metrics_v2.json
step4 → figures/*.png
```

## 快速开始

### 依赖

```bash
pip install numpy opencv-python tqdm matplotlib
```

系统需安装 FFmpeg（用于最终视频编码）。

### 运行

1. 下载 360° 视频放入 `data/`
2. 修改各脚本中的 `VIDEO_PATH` 指向你的视频
3. 按顺序执行：

```bash
cd scripts
python step1_optical_flow.py
python step2_qp_table.py
python step3_jpeg_tile.py
python step4_visualize.py
```

### 配置参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `TILE_ROWS × TILE_COLS` | 5×9 = 45 | 瓦片网格 |
| `FLOW_SCALE` | 0.25 | 光流计算降采样比例 |
| `FRAME_STEP` | 3 | 光流计算帧间隔 |
| `DELTA_QP_MAX` | 6 | 瓦片间最大 QP 差异 |
| `ALPHA / BETA / GAMMA` | 0.35 / 0.45 / 0.20 | 视口/光流/任务 融合权重 |
| `CRF_VALUE` | 23 | FFmpeg 编码质量 |

## 实验结果（Ducati Diavel 骑行视频，30s 段）

| 指标 | Uniform | Flow-Guided |
|---|---|---|
| 视频文件大小 | 57.36 MB | 55.64 MB |
| 平均 PSNR | 32.1 dB | 32.1 dB |
| 码率节省 | — | **3.0%** |

> 注：当前 3% 节省效果偏保守，扩大 ΔQP_max 或延长测试段可进一步放大差异。

## 目标硬件部署

- **开发机**：x86 + Windows/Linux（离线验证）
- **目标平台**：RK3576 ARM 板（Ubuntu 22.04，6 TOPS NPU）
- **相机**：Insta360 X5（2880×1440，USB webcam 模式）
- **推流协议**：RTMP → SRS 服务器

## 参考资料

- 方案设计文档：`光流+视口代理+带宽自适应方案.md`
- 方案对比：`方案对比.md`
