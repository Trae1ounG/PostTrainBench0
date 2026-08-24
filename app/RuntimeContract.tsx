"use client";

import { useEffect, useState } from "react";

type Language = "zh" | "en";
type View = "config" | "workspace" | "instruction";

const configTemplate = `{
  "run_id": "qwen25-3b-agent-run01",
  "paths": {
    "base_model": "/models/Qwen2.5-3B-Instruct",
    "evaluation_data": "/datasets/posttrainbench0/visible200",
    "randopt_source": "/opt/RandOPT",
    "runs_root": "/workspace/posttrainbench0-runs",
    "prompt": "./prompt.txt",
    "starter": "./starter"
  },
  "agent": {
    "harness": "opencode",
    "model": "provider/model-name",
    "cli_path": "/opt/opencode/bin/opencode",
    "credential_files": {},
    "pass_environment": ["OPENAI_API_KEY", "OPENAI_BASE_URL"]
  },
  "evaluation": { "samples_per_task": 200 },
  "runtime": { "hours": 4, "num_gpus": 8 }
}`;

const workspaceTree = `/home/agent/                         ← only writable root
├── prompt.txt                       rendered shared instruction
├── episode.json                     model, tasks, GPU count, deadline
├── timer.sh
├── bin/
│   ├── evaluate                     score one candidate
│   ├── evaluate-batch               score up to eight in parallel
│   ├── results                      completed attempt history
│   └── status                       remaining time + current best
└── starter/
    ├── agent_client.py              score-only socket client
    ├── randopt.py                   optional random-search example
    └── es.py                        optional antithetic ES example

/models/base/                        ← full checkpoint, read-only

Not mounted: evaluator data · scorer · attempts · logs · best · final`;

export default function RuntimeContract({ language }: { language: Language }) {
  const [view, setView] = useState<View>("workspace");
  const [prompt, setPrompt] = useState("Loading instruction template…");
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}prompt.txt`)
      .then((response) => {
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        return response.text();
      })
      .then(setPrompt)
      .catch(() => setPrompt("Instruction template could not be loaded."));
  }, []);

  const content = view === "config" ? configTemplate : view === "workspace" ? workspaceTree : prompt;
  return <figure className="runtime-contract">
    <figcaption><b>{tx("交互框架视图。", "Interactive runtime view.")}</b> {tx("切换查看操作员配置、Agent 实际看到的根目录，以及所有 harness 共用的完整任务指令。", "Switch between the operator configuration, the exact Agent-visible root, and the complete instruction shared by every harness.")}</figcaption>
    <div className="runtime-tabs" role="tablist" aria-label="Runtime contract views">
      {([
        ["config", tx("运行配置", "Run config")],
        ["workspace", tx("可见目录", "Visible files")],
        ["instruction", tx("完整指令", "Full instruction")],
      ] as [View, string][]).map(([id, label]) => <button key={id} role="tab" aria-selected={view === id} className={view === id ? "active" : ""} onClick={() => setView(id)}>{label}</button>)}
    </div>
    <div className="runtime-panel">
      <div className="runtime-panel-head">
        <strong>{view === "config" ? tx("启动前由操作员填写一次", "Filled once by the operator") : view === "workspace" ? tx("Agent 进程中的根目录", "Root visible to the Agent") : tx("prompt.txt 模板（逐字）", "prompt.txt template (verbatim)")}</strong>
        <span>{view === "config" ? tx("路径不会由程序猜测", "No path discovery") : view === "workspace" ? tx("其余路径不挂载", "Everything else is unmounted") : tx("只替换四个占位符", "Only four placeholders are rendered")}</span>
      </div>
      <pre className={view === "instruction" ? "instruction-template" : "runtime-code"}><code>{content}</code></pre>
    </div>
    <div className="runtime-boundary">
      <div><b>{tx("Agent 控制", "Agent controls")}</b><span>{tx("搜索代码、候选定义、评测顺序、任务子集与并行方式", "Search code, candidates, evaluation order, task subsets, and parallelism")}</span></div>
      <div><b>{tx("系统控制", "System controls")}</b><span>{tx("只读基础权重、隐藏数据与评分、永久 attempt、最高点保留与最终重放", "Read-only base weights, hidden data/scoring, permanent attempts, best retention, and replay")}</span></div>
    </div>
  </figure>;
}
