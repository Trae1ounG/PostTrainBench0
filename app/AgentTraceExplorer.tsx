"use client";

import { useState } from "react";
import type { CSSProperties } from "react";

type Language = "zh" | "en";
type Stage = {
  titleZh: string;
  titleEn: string;
  actionZh: string;
  actionEn: string;
  policy: string;
  code: string;
  score: string;
  nextZh: string;
  nextEn: string;
};

const traces: Record<string, { label: string; color: string; summaryZh: string; summaryEn: string; stages: Stage[] }> = {
  gpt56: {
    label: "GPT-5.6 xhigh · Codex",
    color: "#2767cc",
    summaryZh: "先用成对探测定位方向，再沿同一方向做尺度搜索；平台期后改用弱任务筛选，并把新方向组合进当前最好点。",
    summaryEn: "It first located directions with antithetic probes, line-searched one direction, then switched to weak-task screening and composed promoted directions after a plateau.",
    stages: [
      { titleZh: "建立基线", titleEn: "Establish baseline", actionZh: "完整评测基础模型，保存 41.08 作为所有后续决策的参照。", actionEn: "Evaluate the base on all seven tasks and retain 41.08 as the reference for every later decision.", policy: "full suite · base center", code: "bin/evaluate candidates/base.json", score: "41.08", nextZh: "先确认随机方向是否有可测信号。", nextEn: "Check whether random directions carry measurable signal." },
      { titleZh: "正反方向探测", titleEn: "Antithetic probes", actionZh: "测试四组成对方向，发现 seed 107420369 的负方向优于正方向。", actionEn: "Test four positive/negative pairs and find that the negative side of seed 107420369 is promising.", policy: "4 pairs · ±0.0005 · full suite", code: "for seed in seeds:\n  evaluate(+sigma, -sigma)", score: "42.84", nextZh: "复用这个方向，不再重新抽样。", nextEn: "Reuse this direction rather than drawing a new one." },
      { titleZh: "沿方向搜索尺度", titleEn: "Line search scale", actionZh: "保持 seed 不变，把负尺度从 −0.0005 逐步扩大到约 −0.0015。", actionEn: "Hold the seed fixed and increase the negative scale from −0.0005 toward roughly −0.0015.", policy: "same direction · six scales · full suite", code: "for s in [-.0006, -.0007, -.0010, -.0012, -.0015]:\n  evaluate(seed, s)", score: "47.24", nextZh: "围绕当前最好点加入小修正方向。", nextEn: "Add small corrective directions around the incumbent." },
      { titleZh: "组合方向", titleEn: "Compose directions", actionZh: "把新的正反探测结果加入已有中心，从二项组合扩展到五项组合。", actionEn: "Add newly tested directions to the incumbent, growing from two-term to five-term programs.", policy: "stateful center · full-suite promotion", code: "candidate = incumbent + coefficient * direction", score: "48.83", nextZh: "完整评测进入平台期，转向定位薄弱任务。", nextEn: "The full-suite curve plateaus, so diagnose weak tasks." },
      { titleZh: "任务子集筛选", titleEn: "Task-subset screening", actionZh: "先在薄弱任务上快速淘汰方向，只有通过筛选的候选才升级到七任务评测。", actionEn: "Screen directions on weak tasks first and promote only survivors to the full seven-task suite.", policy: "weak-task proxy → full-suite promotion", code: "probe = evaluate(candidate, weak_tasks)\nif probe > gate: evaluate(candidate, all_tasks)", score: "53.19", nextZh: "保留完整评测中的最高点。", nextEn: "Retain the best candidate with a complete evaluation." },
    ],
  },
  minimax: {
    label: "MiniMax M2.7 · OpenCode",
    color: "#d85a32",
    summaryZh: "少量宽搜后迅速切换为成对 ES；它的关键收益来自把正反分数差转换为有状态的中心更新。",
    summaryEn: "After a small broad sweep, it quickly switched to paired ES; its main gain came from turning score differences into a stateful center update.",
    stages: [
      { titleZh: "宽尺度抽样", titleEn: "Broad scale sweep", actionZh: "围绕基础模型测试 8 个候选，尺度覆盖 0.0005–0.005。", actionEn: "Test eight candidates around the base across scales from 0.0005 to 0.005.", policy: "8 independent candidates · full suite", code: "scales = [.0005, .001, .002, .003, .005]", score: "44.38", nextZh: "用成对方向估计更可靠的更新。", nextEn: "Use paired directions for a more informative update." },
      { titleZh: "第一次 ES 更新", titleEn: "First ES update", actionZh: "评测 4 组正反方向，用分数差计算系数并合成两项中心。", actionEn: "Evaluate four antithetic pairs, turn score differences into coefficients, and build a two-term center.", policy: "4 pairs · σ=0.0005 · stateful", code: "coef = lr * (score_plus - score_minus)\ncenter += coef * direction", score: "48.54", nextZh: "在新中心附近再做一轮，而不是回到基础模型。", nextEn: "Probe around the new center instead of returning to the base." },
      { titleZh: "第二次 ES 更新", titleEn: "Second ES update", actionZh: "复用当前中心继续成对探测，得到最终的小幅提升。", actionEn: "Continue antithetic probing from the incumbent center for a final incremental gain.", policy: "centered pairs · full suite", code: "next_center = update(current_center, paired_scores)", score: "48.70", nextZh: "时间结束时由控制器保留最高完整分。", nextEn: "At timeout, the controller keeps the best complete score." },
    ],
  },
  glm: {
    label: "GLM-5.1 · OpenCode",
    color: "#3e9b70",
    summaryZh: "先做大范围随机尺度试验，排除破坏性步长；随后围绕单个有效 seed 做尺度细化，再用 ES 修正。",
    summaryEn: "It began with a wide random scale sweep to reject destructive steps, refined one useful seed, and then used ES-style corrections.",
    stages: [
      { titleZh: "大范围随机试验", titleEn: "Broad random sweep", actionZh: "测试 48 个候选，覆盖正负 0.0002–0.01；大尺度普遍破坏性能。", actionEn: "Test 48 candidates across positive and negative scales from 0.0002 to 0.01; large steps are usually destructive.", policy: "48 candidates · mixed scales", code: "for seed, scale in broad_grid:\n  evaluate(seed, scale)", score: "44.51", nextZh: "围绕最好 seed 缩小范围。", nextEn: "Narrow the search around the best seed." },
      { titleZh: "单方向尺度细化", titleEn: "Single-direction refinement", actionZh: "固定较好的方向，做更密的尺度扫描并找到约 0.00095 的局部点。", actionEn: "Fix a useful direction and run a denser scale sweep, locating a local point near 0.00095.", policy: "fixed seed · fine scale grid", code: "candidate(seed=best_seed, scale=.00095)", score: "45.24", nextZh: "尝试用分数差加入修正方向。", nextEn: "Use score differences to add a corrective direction." },
      { titleZh: "ES 修正与组合", titleEn: "ES correction and composition", actionZh: "加入第二个方向后升至 46.67；继续加入第三、第四项没有再提高。", actionEn: "Adding a second direction reaches 46.67; third and fourth terms do not improve it further.", policy: "incumbent + corrective terms", code: "best = base + term_1 + correction_2", score: "46.67", nextZh: "保留两项组合，停止无收益扩展。", nextEn: "Keep the two-term candidate and stop unproductive expansion." },
    ],
  },
};

export default function AgentTraceExplorer({ language }: { language: Language }) {
  const [agent, setAgent] = useState("gpt56");
  const [stageIndex, setStageIndex] = useState(0);
  const trace = traces[agent];
  const stage = trace.stages[stageIndex];
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;

  const selectAgent = (id: string) => { setAgent(id); setStageIndex(0); };
  return <figure className="agent-trace-explorer">
    <figcaption><b>{tx("交互 trace 阅读器。", "Interactive trace reader.")}</b> {tx("选择 Agent，再逐步查看它如何写搜索逻辑、改变采样策略，以及分数反馈如何触发下一步。代码是根据完整 trace 还原的结构，不是逐字转录。", "Choose an Agent and step through how it wrote search logic, changed its sampling policy, and reacted to score feedback. Code is a trace-derived structural sketch, not a verbatim transcript.")}</figcaption>
    <div className="trace-agent-tabs" role="tablist">
      {Object.entries(traces).map(([id, item]) => <button key={id} role="tab" aria-selected={agent === id} className={agent === id ? "active" : ""} style={{ "--trace-color": item.color } as CSSProperties} onClick={() => selectAgent(id)}>{item.label}</button>)}
    </div>
    <p className="trace-summary">{language === "zh" ? trace.summaryZh : trace.summaryEn}</p>
    <div className="trace-steps" style={{ "--trace-color": trace.color } as CSSProperties}>
      {trace.stages.map((item, index) => <button key={item.titleEn} className={stageIndex === index ? "active" : ""} onClick={() => setStageIndex(index)}><i>{index + 1}</i><span>{language === "zh" ? item.titleZh : item.titleEn}</span><b>{item.score}</b></button>)}
    </div>
    <div className="trace-detail">
      <div className="trace-decision"><span>{tx("研究决策", "Research decision")}</span><h4>{language === "zh" ? stage.titleZh : stage.titleEn}</h4><p>{language === "zh" ? stage.actionZh : stage.actionEn}</p><dl><div><dt>{tx("采样策略", "Sampling policy")}</dt><dd>{stage.policy}</dd></div><div><dt>{tx("当时最好分", "Running best")}</dt><dd>{stage.score}</dd></div></dl></div>
      <pre className="trace-code"><code>{stage.code}</code></pre>
      <div className="trace-next"><span>{tx("反馈如何改变下一步", "How feedback changed the next move")}</span><p>{language === "zh" ? stage.nextZh : stage.nextEn}</p></div>
    </div>
  </figure>;
}
