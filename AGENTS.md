# AGENTS.md

## Git 工作约定

- 用户要求提交后，**自动用 SSH 推送**：`git commit` 之后执行 `git push`（remote 已是 SSH：`git@github.com:BodNoone/agentic-gts.git`）。
- 只暂存本次任务相关的文件；不要纳入未跟踪的大文件与探针脚本（`godview.jpg`、`docs/pipeline_flow.jpg`、`_probe_runner.py`）。
- 注意 `agentic_gts/agent/ground.py` 常只有 CRLF 行尾差异（无实质 diff），不要误提交。
- 提交信息沿用仓库风格：单行长摘要，英文，说明改动与动机（常引用 user report / user direction）。

## 测试

用项目 conda 环境的 Python（base 环境无 pytest）：

```powershell
& "C:\Users\WeiFang\miniconda3\envs\agentic-gts\python.exe" -m pytest tests/ -q
```

单文件示例：`... -m pytest tests/test_mask_refine.py -q`
