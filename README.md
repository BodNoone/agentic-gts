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

# 带真值评测
python -m agentic_gts.cli run --point-cloud room.ply \
    --gt gt_boxes.json --edge-thr 0.05 --out runs/room1
```

### 2. 接入 Qwen3-VL 裁判

启动一个 OpenAI 兼容服务（vLLM / SGLang / DashScope 均可）：

```bash
# 例：vLLM
vllm serve Qwen/Qwen3-VL-8B-Instruct --port 8000
```

然后：

```bash
python -m agentic_gts.cli run --point-cloud room.ply \
    --vlm qwen --vlm-base http://127.0.0.1:8000/v1 --out runs/room1
```

不配置 VLM 时自动使用规则降级模式（mock），管线仍可运行但 grounding 不产出 box（无输入 box、无 fallback）——这也是可靠性下限基线。

### 3. 局部细化的 SAM2 mask（可选但推荐）

```bash
python -m agentic_gts.cli run --point-cloud room.ply \
    --vlm qwen --vlm-base http://127.0.0.1:8000/v1 \
    --sam-checkpoint sam2.1_hiera_base_plus.pt --sam-model-cfg sam2.1_hiera_b+.yaml \
    --out runs/room1
```

### 4. 生成合成测试数据

```bash
python -m agentic_gts.cli synth --seed 42 --out runs/synth
# 产出 points.npy / gt_boxes.json / corrupted_boxes.json
```

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
│                         提供 --gt 时真值框以蓝色虚线叠加，可直观对比偏差）
├── cloud_with_boxes.ply  点云 + box 线框合并 PLY（CloudCompare/MeshLab 直接打开做 3D 检查）
├── agent_report.json     agent 决策记录（issue → 动作 → 结果）
└── eval.json             分阶段评测（提供 --gt 时）
```

**输出坐标系与输入点云一致**：管线内部的地面对齐（调平/归零）在写出前已逆映射回原始坐标，`boxes.json` / `cloud_with_boxes.ply` 可直接叠在原始点云上使用。

### 3D 交互查看

```bash
# 打开 Open3D 窗口：点云 + 3D 线框框（绿=high / 黄=mid / 红=low，蓝=真值）
python -m agentic_gts.cli view --point-cloud room.ply --boxes runs/room1/boxes.json

# 或直接用任意点云软件打开合并 PLY
# CloudCompare runs/room1/cloud_with_boxes.ply
```

## 评测指标

按验收标准实现：**贴边准确率** = 预测 box 边与匹配真值 box 边的垂直误差 < 阈值（默认 5cm，`--edge-thr` 可调）的边占比。同时报告 recall / precision / mean / p90 边误差。

## 代码结构

```
agentic_gts/
├── core/models.py        OrientedBox / Scene / Issue 数据模型
├── synth/generator.py    合成机房生成器（含四类噪声注入）
├── segment/orientation.py 阶段0：yaw 估计 + 布局 bootstrap（footprint / z_top）
├── tools/geometry.py     几何工具集（支撑度）
├── agent/judge.py        VLM 裁判（Qwen3-VL 接口 + mock 降级）
├── agent/ground.py       阶段G：全局 nadir VLM 2D grounding
├── agent/mask_refine.py  阶段C：三视角局部细化（SAM2 mask / 切分 / 高度修正）
├── agent/loop.py         阶段C：agent 修复循环（诊断→动作→验证→回滚）
├── eval/metrics.py       贴边准确率评测
├── output/render.py      SVG/PNG 布局图
├── output/visualize.py   点云+框联合可视化（2D叠加 / 3D交互 / PLY导出）
├── pipeline.py           全流程编排
└── cli.py                命令行入口（run / synth / diagnose / view / report）
tests/                    单元 + 端到端测试
docs/                     设计方案文档
```

## 测试

```bash
python -m pytest tests/ -q
# 85 passed
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
