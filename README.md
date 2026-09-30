# agentic-gts

数字机房布局图自动化生成 —— Agent 化 3DGS 后处理系统。

从 3DGS 重建导出的点云出发，自动生成设备（机柜等）2D 布局图。**VLM 做全局 grounding 与判别、几何算法做精确测量**：阶段0 估计朝向并 bootstrap 设备布局，阶段G 由 VLM 在俯视图上圈出所有设备结构，阶段C 对每个 box 做三视角局部细化（切分/厚度/高度修正）。无任何初始 box 输入。

## 核心设计

```
点云 (3DGS 导出)
    ↓
阶段0  朝向估计 + 布局 bootstrap（垂直面滤波 + 边界去墙，
       产出 device_footprint 设备外框与 z_top 设备顶高）
    ↓
阶段G  全局 nadir VLM 2D grounding（VLM 在俯视图上圈出每个设备结构，
       几何按点支撑拟合全深度 box —— 唯一的 box 生产者，无任何输入 box）
    ↓
阶段C  每盒局部细化（三视角渲染 → VLM 质量判断 + bbox → SAM2 mask →
       反投影 → 沿排切分 / 厚度修正 / 锚定高度修正）
    ↓
阶段D  行补全兜底（几何纯点支撑探针：行内空隙与行端步进探测，
       补回 VLM 漏检的机柜，标记 LOW 置信度待人工复核）
    ↓
布局图 (SVG/PNG) + boxes.json + 置信度标记 + 评测报告
```

**关键原则**：VLM 只做判别（"这是一个柜还是两个柜？"），精确坐标永远由几何算法产出。

**任意朝向支持**：管线不假设点云横平竖直。阶段0 自动估计设备行方向 yaw（局部边缘方向直方图 + 行带质量评分，机房曼哈顿结构假设），后续所有阶段统一使用。旋转 15/30/60° 的场景 yaw 估计误差 < 1°。若你已知朝向，也可在 `run_pipeline` 的 `opts` 传 `yaw` 跳过估计。

### 设计机制

- **基础模型当裁判**而非特征提取器 → VLM 判别接口跨机房免重训
- **验证-回滚闭环** → 每次修复后重检支撑度/尺寸/重叠，不通过即回滚
- **锚定高度估计** → 从地面向上按密度连通遍历，天然分离柜体与上方桥架/线缆浮层

## 安装

```bash
conda create -n agentic-gts python=3.10 -y
conda activate agentic-gts
pip install -r requirements.txt
```

## 使用

### 1. 处理你自己的点云

```bash
# 从点云直接跑（box 全部来自 VLM grounding，无需任何初始输入）
python -m agentic_gts.cli run --point-cloud room.ply --out runs/room1

# 评测与合成数据仅供内部测试工具使用，见下文
```

### 2. 使用本地 VLM

```bash
python -m agentic_gts.cli run --point-cloud room.ply \
    --vlm local --vlm-model Qwen/Qwen3-VL-8B-Instruct --out runs/room1
```

默认使用本地 VLM（`--vlm local`）。通过 `--vlm-model` 指定本地模型目录；未配置可用模型时不会自动加载替代模型，测试中应显式注入本地调用 double。grounding 失败时场景保持为空，没有输入 box 或 fallback。

### 3. 局部细化的 SAM2 mask（可选但推荐）

```bash
python -m agentic_gts.cli run --point-cloud room.ply \
    --vlm local --vlm-model /models/qwen3-vl \
    --sam-checkpoint sam2.1_hiera_base_plus.pt --sam-model-cfg sam2.1_hiera_b+.yaml \
    --out runs/room1
```

### 4. 内部测试与评测工具

合成数据生成器、真值数据和评测指标仍保留在 Python 内部测试工具中，供
`tests/` 和开发验证使用；它们不再是生产 CLI 子命令或生产后端。

## 输入输出格式

**输入点云**：`.ply` / `.pcd`（open3d 可读）或 `.npy`（Nx3 float，单位米，z 向上，地面 z≈0）。

**box JSON**（输入与输出同格式）：

```json
[{
  "center": [1.2, 3.4, 1.0],
  "size": [0.6, 1.1, 2.0],
  "yaw": 0.0,
  "device_type": "rack",
  "confidence": "high",
  "source": "agent_fix",
  "row_id": 0
}]
```

**输出目录**：

```
runs/xxx/
├── boxes.json            最终 box（带置信度：high 自动接受 / mid / low 建议人工复核）
├── layout.svg            矢量布局图（按置信度着色）
├── layout.png            布局预览图
├── overlay.png           点云 + 检测框叠加图（点云按高度着色；框按置信度着色；
│                         真值对比由内部评测工具处理）
├── cloud_with_boxes.ply  点云 + box 线框合并 PLY（CloudCompare/MeshLab 直接打开做 3D 检查）
├── agent_report.json     agent 决策记录（issue → 动作 → 结果）
└── eval.json             分阶段评测（由内部评测工具生成）
```

**输出坐标系与输入点云一致**：管线内部的地面对齐（调平/归零）在写出前已逆映射回原始坐标，`boxes.json` / `cloud_with_boxes.ply` 可直接叠在原始点云上使用。

## 评测指标

按验收标准实现：**贴边准确率** = 预测 box 边与匹配真值 box 边的垂直误差 < 阈值的边占比。同时报告 recall / precision / mean / p90 边误差。该指标和阈值配置保留给内部评测工具，不属于生产 CLI 参数。

## 代码结构

```
agentic_gts/
├── core/models.py        OrientedBox / Scene 数据模型
├── synth/generator.py    合成机房生成器（含四类噪声注入）
├── segment/orientation.py 阶段0：yaw 估计 + 布局 bootstrap（footprint / z_top）
├── tools/geometry.py     几何工具集（支撑度）
├── agent/judge.py        VLM 裁判（local 后端；Qwen 兼容接口保留为内部后端）
├── agent/ground.py       阶段G：全局 nadir VLM 2D grounding
├── agent/mask_refine.py  阶段C：三视角局部细化（SAM2 mask / 切分 / 高度修正）
├── agent/loop.py         阶段C：agent 修复循环（诊断→动作→验证→回滚）
├── eval/metrics.py       贴边准确率评测
├── output/render.py      SVG/PNG 布局图
├── output/visualize.py   点云+框联合可视化（2D叠加 / PLY导出）
├── pipeline.py           全流程编排
└── cli.py                命令行入口（run）
tests/                    单元 + 端到端测试
docs/                     设计方案文档
```

## 测试

```bash
python -m pytest tests/ -q
# 合成数据和评测测试仍由 tests/ 覆盖
```

## 与真实 3DGS pipeline 对接

1. 3DGS 重建后导出点云（Gaussian 中心即可）为 PLY/NPY；管线直接从裸点云跑，box 全部来自 VLM grounding。
2. 设备标称尺寸可选：在 `run_pipeline` 的 `opts` 里传 `width_unit`（默认 0.6m）、`depth`、`height`；没有标称尺寸时系统按点云支撑自适应。
3. 输出 `boxes.json` 中 `confidence=low` 的项送人工复核；人工修正结果与 agent 决策记录一并留存，作为后续训练 3D 检测模型的数据（数据飞轮）。

## 已知限制

- 布局假设设备按行摆放（机房通用），非行结构场景（散放设备）效果会退化。
- grounding 失败（VLM 不可用 / 无 region 通过点支撑守卫）时场景保持为空，没有 fallback box。
- 动态场景 / 多层机房未覆盖。
- 尚无设备类型校验（柱子/墙体有可能被 VLM 误圈为机柜）。
