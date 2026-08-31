# agentic-gts

数字机房布局图自动化生成 —— Agent 化 3DGS 后处理系统。

从 3DGS 重建导出的点云出发，自动生成设备（机柜等）2D 布局图。用**确定性几何规则打底、VLM 判别裁决兜底**的 agent 循环，替代人工返工中的四类修正操作：box 偏大、漏检、错检、联排机柜粘连。

## 核心设计

```
点云 (3DGS 导出)
    ↓
阶段A  粗分割（superpoint 过分割 + 行结构合并，高召回）    [可选：也可直接输入已有检测 box]
    ↓
阶段B  确定性规则（墙面过滤 / 尺寸 / 支撑度 / 行补全 / 联排切分）
    ↓
阶段C  Agent 修复循环（取证 → VLM 诊断 → 离散动作 → 几何工具执行 → 验证/回滚）
    ↓
布局图 (SVG/PNG) + boxes.json + 置信度标记 + 评测报告
```

**关键原则**：VLM 只做判别（"这是一个柜还是两个柜？"），精确坐标永远由几何算法产出。

**任意朝向支持**：管线不假设点云横平竖直。阶段0 自动估计设备行方向 yaw（局部边缘方向直方图 + 行带质量评分，机房曼哈顿结构假设），后续所有阶段统一使用。旋转 15/30/60° 的场景 yaw 估计误差 < 1°。若你已知朝向，也可在 `run_pipeline` 的 `opts` 传 `yaw` 跳过估计。

### 借鉴 FoundObj (ICML 2026) 的机制

- **中心场完整性验证** → `center_field_clusters`：沿行轴统计侧面密度峰，k+1 个峰 = k 个机柜，自动触发联排切分
- **基础模型当裁判**而非特征提取器 → VLM 判别接口跨机房免重训
- **验证-回滚闭环** → 每次修复后重检支撑度/尺寸/重叠，不通过即回滚

## 实测效果（合成数据，边误差阈值 5cm）

输入为带四类噪声的模拟检测结果（对应真实 GS 分割输出）：

| 阶段 | edge_acc | mean_err | recall | precision |
|---|---:|---:|---:|---:|
| 输入（含噪声） | 79.4% | 5.9cm | 81.0% | 100% |
| 阶段B 规则后 | 90.5% | 1.7cm | **100%** | 95.5% |
| 阶段C Agent 后 | **96.4%** | **1.7cm** | 100% | 95.5% |

多随机种子（3/7/11/19）：edge_acc 92–94%，recall 96–100%，precision 96%。

## 安装

```bash
conda create -n agentic-gts python=3.10 -y
conda activate agentic-gts
pip install -r requirements.txt
```

## 使用

### 1. 一键 Demo（合成数据 → 全流程 → 评测）

```bash
python -m agentic_gts.cli demo --out runs/demo
```

### 2. 处理你自己的点云

```bash
# 从头跑（含粗分割）
python -m agentic_gts.cli run --point-cloud room.ply --out runs/room1

# 已有初始检测 box（你现有 GS 分割 pipeline 的输出）
python -m agentic_gts.cli run --point-cloud room.ply --boxes init_boxes.json --out runs/room1

# 带真值评测
python -m agentic_gts.cli run --point-cloud room.ply --boxes init_boxes.json \
    --gt gt_boxes.json --edge-thr 0.05 --out runs/room1
```

### 3. 接入 Qwen3-VL 裁判

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

不配置 VLM 时自动使用规则降级模式（mock），整个管线仍可运行——这也是可靠性下限基线。

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
├── segment/coarse.py     阶段A：superpoint 粗分割
├── rules/rules.py        阶段B：确定性规则
├── tools/geometry.py     几何工具集（fit_box / split / 中心场 / 行结构 / 支撑度）
├── agent/judge.py        VLM 裁判（Qwen3-VL 接口 + mock 降级）
├── agent/loop.py         阶段C：agent 修复循环（诊断→动作→验证→回滚）
├── eval/metrics.py       贴边准确率评测
├── output/render.py      SVG/PNG 布局图
├── output/visualize.py   点云+框联合可视化（2D叠加 / 3D交互 / PLY导出）
├── pipeline.py           全流程编排
└── cli.py                命令行入口（demo / run / synth / view）
tests/test_pipeline.py    单元 + 端到端测试（7 项）
docs/                     设计方案文档
```

## 测试

```bash
python tests/test_pipeline.py
# 7/7 tests passed
```

## 与真实 3DGS pipeline 对接

1. 3DGS 重建后导出点云（Gaussian 中心即可）为 PLY/NPY。
2. 若已有 GS 分割结果，把 3D BBox 转成上述 JSON 作为 `--boxes` 输入（推荐，跳过粗分割）。
3. 设备标称尺寸可选：在 `run_pipeline` 的 `opts` 里传 `width_unit`（默认 0.6m）、`depth`、`height`；没有标称尺寸时系统按点云支撑自适应。
4. 输出 `boxes.json` 中 `confidence=low` 的项送人工复核；人工修正结果与 agent 决策记录一并留存，作为后续训练 3D 检测模型的数据（数据飞轮）。

## 已知限制

- 粗分割（阶段A）在本版本中主要产出行级候选，依赖阶段B切分为单柜；如已有检测 box 建议直接走 `--boxes` 输入路径。
- 布局假设设备按行摆放（机房通用），非行结构场景（散放设备）需调整 `row_structure` 容差。
- 动态场景 / 多层机房未覆盖。
- VLM 裁判当前只在 merged / false-positive 两类 issue 上介入；证据图为俯视密度图，可扩展接入 3DGS 渲染视图。
