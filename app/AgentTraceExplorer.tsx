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

type Trace = {
  label: string;
  color: string;
  target: string;
  outcomeZh: string;
  outcomeEn: string;
  summaryZh: string;
  summaryEn: string;
  conclusionZh: string;
  conclusionEn: string;
  stages: Stage[];
};

const traces: Record<string, Trace> = {
  gpt56: {
    label: "GPT-5.6 xhigh · Codex",
    color: "#2767cc",
    target: "Qwen3-4B-Base",
    outcomeZh: "41.08 → 53.19（该设置的最高运行）",
    outcomeEn: "41.08 → 53.19 (best run for this setting)",
    summaryZh: "先用成对探测定位方向，再沿同一方向做尺度搜索；平台期后改用弱任务筛选，并把新方向组合进当前最好点。",
    summaryEn: "It first located directions with antithetic probes, line-searched one direction, then switched to weak-task screening and composed promoted directions after a plateau.",
    conclusionZh: "这条轨迹展示了完整的“探测—复用—诊断—组合”循环，也达到了全部 Qwen3-4B 运行中的最高分；但同一设置的另一次运行只有 43.29，因此高上限并没有转化为稳定结果。",
    conclusionEn: "This trace completes a probe–reuse–diagnose–compose loop and reaches the highest score among all Qwen3-4B runs. Another repeat of the same setting reaches only 43.29, so the high ceiling does not translate into stable performance.",
    stages: [
      { titleZh: "建立基线", titleEn: "Establish baseline", actionZh: "完整评测基础模型，保存 41.08 作为所有后续决策的参照。", actionEn: "Evaluate the base on all seven tasks and retain 41.08 as the reference for every later decision.", policy: "full suite · base center", code: "bin/evaluate candidates/base.json", score: "41.08", nextZh: "先确认随机方向是否有可测信号。", nextEn: "Check whether random directions carry measurable signal." },
      { titleZh: "正反方向探测", titleEn: "Antithetic probes", actionZh: "测试四组成对方向，发现 seed 107420369 的负方向优于正方向。", actionEn: "Test four positive/negative pairs and find that the negative side of seed 107420369 is promising.", policy: "4 pairs · ±0.0005 · full suite", code: "for seed in seeds:\n  evaluate(+sigma, -sigma)", score: "42.84", nextZh: "复用这个方向，不再重新抽样。", nextEn: "Reuse this direction rather than drawing a new one." },
      { titleZh: "沿方向搜索尺度", titleEn: "Line search scale", actionZh: "保持 seed 不变，把负尺度从 −0.0005 逐步扩大到约 −0.0015。", actionEn: "Hold the seed fixed and increase the negative scale from −0.0005 toward roughly −0.0015.", policy: "same direction · six scales · full suite", code: "for s in [-.0006, -.0007, -.0010, -.0012, -.0015]:\n  evaluate(seed, s)", score: "47.24", nextZh: "围绕当前最好点加入小修正方向。", nextEn: "Add small corrective directions around the incumbent." },
      { titleZh: "组合方向", titleEn: "Compose directions", actionZh: "把新的正反探测结果加入已有中心，从二项组合扩展到五项组合。", actionEn: "Add newly tested directions to the incumbent, growing from two-term to five-term programs.", policy: "stateful center · full-suite promotion", code: "candidate = incumbent + coefficient * direction", score: "48.83", nextZh: "完整评测进入平台期，转向定位薄弱任务。", nextEn: "The full-suite curve plateaus, so diagnose weak tasks." },
      { titleZh: "任务子集筛选", titleEn: "Task-subset screening", actionZh: "先在薄弱任务上快速淘汰方向，只有通过筛选的候选才升级到七任务评测。", actionEn: "Screen directions on weak tasks first and promote only survivors to the full seven-task suite.", policy: "weak-task proxy → full-suite promotion", code: "probe = evaluate(candidate, weak_tasks)\nif probe > gate: evaluate(candidate, all_tasks)", score: "53.19", nextZh: "保留完整评测中的最高点。", nextEn: "Retain the best candidate with a complete evaluation." },
    ],
  },
  kimi: {
    label: "Kimi K2.6 · OpenCode",
    color: "#7957c8",
    target: "Qwen2.5-3B-Instruct",
    outcomeZh: "44.08 → 49.25；488 次完整七任务评测",
    outcomeEn: "44.08 → 49.25; 488 complete seven-task evaluations",
    summaryZh: "先筛选单方向，再把有效方向组成两项和三项候选；随后围绕已有组合继续细化尺度，而不是每轮回到基础模型重新抽样。",
    summaryEn: "It screened single directions, composed useful ones into two- and three-term candidates, and then refined scales around the incumbent instead of restarting from the base model.",
    conclusionZh: "Kimi 的三次 Qwen2.5-3B 运行分别达到 47.51、47.51 和 49.25。最高运行中，129 个单项候选先确定方向，349 个多项候选再完成组合；最好三项 checkpoint 比最好单项高 2.65 分。这说明它确实利用了历史反馈，但最终收益仍明显依赖本次抽到的方向。",
    conclusionEn: "Kimi's three Qwen2.5-3B runs reach 47.51, 47.51, and 49.25. In the best run, 129 one-term candidates identify directions before 349 multi-term candidates compose them; the best three-term checkpoint beats the best one-term checkpoint by 2.65 points. It clearly uses prior feedback, while the final gain still depends strongly on the sampled directions.",
    stages: [
      { titleZh: "建立基线", titleEn: "Establish baseline", actionZh: "完整评测基础模型，将 44.08 作为后续单项与组合候选的共同参照。", actionEn: "Evaluate the base on all seven tasks and use 44.08 as the common reference for both single-term and composed candidates.", policy: "full suite · base center", code: "base_score = evaluate(base, all_tasks)", score: "44.08", nextZh: "先确认安全尺度，并收集可复用方向。", nextEn: "Identify safe scales and collect reusable directions." },
      { titleZh: "筛选单方向", titleEn: "Screen single directions", actionZh: "跨多个 seed 和尺度评测 129 个单项候选；大步长被淘汰，最好单项达到 46.60。", actionEn: "Evaluate 129 one-term candidates across seeds and scales; reject destructive large steps and retain a best single term at 46.60.", policy: "129 one-term candidates · full suite", code: "for seed, scale in single_term_pool:\n  score(seed, scale)", score: "46.60", nextZh: "不停止在最好单项，测试方向之间能否互补。", nextEn: "Do not stop at the best single term; test whether directions complement one another." },
      { titleZh: "构造两项组合", titleEn: "Build two-term candidates", actionZh: "以最好方向为中心加入第二个扰动，最好两项 checkpoint 达到 47.82。", actionEn: "Add a second perturbation to the best direction; the strongest two-term checkpoint reaches 47.82.", policy: "incumbent + second direction", code: "candidate = base + term_1 + term_2", score: "47.82", nextZh: "围绕组合中心继续调整系数。", nextEn: "Refine coefficients around the composed center." },
      { titleZh: "局部尺度细化", titleEn: "Local scale refinement", actionZh: "围绕已有组合做密集的小尺度搜索，完整分数首次越过 48 分。", actionEn: "Run a dense small-scale search around the incumbent composition, pushing the complete score above 48 for the first time.", policy: "local grid around incumbent", code: "for delta in local_grid:\n  evaluate(incumbent + delta)", score: "48.01", nextZh: "加入第三个互补方向并做完整复验。", nextEn: "Add a third complementary direction and run a complete replay." },
      { titleZh: "三项最终候选", titleEn: "Three-term incumbent", actionZh: "把第三个有效方向加入当前最好点；全新实例重放后，七任务均分为 49.25。", actionEn: "Add a third useful direction to the incumbent; a fresh-instance replay yields a seven-task mean of 49.25.", policy: "3 terms · fresh-instance replay", code: "final = base + term_1 + term_2 + term_3", score: "49.25", nextZh: "控制器保留最高完整评测 checkpoint。", nextEn: "The controller retains the best fully evaluated checkpoint." },
    ],
  },
  opus: {
    label: "Claude Opus 4.8 high · Cursor",
    color: "#b54f70",
    target: "Qwen2.5-3B-Instruct",
    outcomeZh: "44.08 → 47.13；402 次完整七任务评测",
    outcomeEn: "44.08 → 47.13; 402 complete seven-task evaluations",
    summaryZh: "先广泛筛选单方向，再把小幅有效方向逐项加入当前中心；代表运行的收益主要来自九项有状态组合，而不是某个特别强的单一扰动。",
    summaryEn: "It broadly screened single directions and then added small useful directions to the current center one at a time. The representative run gains mainly from a stateful nine-term composition rather than one unusually strong perturbation.",
    conclusionZh: "Opus 展示了较强的长程状态维护：最好单项只有 44.56，组合到第八项和第九项后才升至 46.63 和 47.13。它的另外两次运行分别停在 46.10 和 46.04，且最好点都是单项候选。也就是说，Opus 能写出有效的组合搜索，但这种行为并不会在每次运行中稳定出现。",
    conclusionEn: "Opus demonstrates strong long-horizon state maintenance: its best single term is only 44.56, while the eighth and ninth terms raise the score to 46.63 and 47.13. Its other two runs stop at 46.10 and 46.04, both with one-term incumbents. Opus can therefore implement effective compositional search, but that behavior does not emerge consistently in every run.",
    stages: [
      { titleZh: "建立基线", titleEn: "Establish baseline", actionZh: "从同一个 44.08 基础 checkpoint 开始，先完整验证评测和重放链路。", actionEn: "Start from the same 44.08 base checkpoint and verify the complete evaluation and replay path.", policy: "full suite · base center", code: "base_score = evaluate(base, all_tasks)", score: "44.08", nextZh: "用多组 seed 和正负尺度扫描局部空间。", nextEn: "Scan the local space with multiple seeds and signed scales." },
      { titleZh: "单项方向扫描", titleEn: "Single-term sweep", actionZh: "评测 16 个单项候选；最好单方向只达到 44.56，没有出现足以单独解释最终收益的强方向。", actionEn: "Evaluate 16 one-term candidates; the best reaches only 44.56, leaving no single direction strong enough to explain the final gain.", policy: "16 one-term candidates · signed scales", code: "pool = evaluate(single_terms)\nincumbent = max(pool)", score: "44.56", nextZh: "保存微弱有效方向，改为逐项组合。", nextEn: "Retain weakly useful directions and compose them incrementally." },
      { titleZh: "逐项移动中心", titleEn: "Move the center incrementally", actionZh: "每加入一个方向都从新中心重新评测；四项和五项组合依次达到 45.29 与 45.52。", actionEn: "Re-evaluate from the new center after each added direction; four- and five-term compositions reach 45.29 and 45.52.", policy: "stateful additive composition", code: "if score(proposal) > score(center):\n  center = proposal", score: "45.52", nextZh: "继续保留能提高联合分数的小修正。", nextEn: "Continue retaining small corrections that improve the joint score." },
      { titleZh: "八项组合", titleEn: "Eight-term composition", actionZh: "持续累积小方向，第八项使均分升至 46.63，超过所有单项候选。", actionEn: "Continue accumulating small directions; the eighth term raises the mean to 46.63, above every single-term candidate.", policy: "8-term incumbent · full suite", code: "center = base + sum(accepted_terms[:8])", score: "46.63", nextZh: "验证最后一个候选是否仍能提供互补收益。", nextEn: "Test whether one final candidate adds complementary gain." },
      { titleZh: "九项最终候选", titleEn: "Nine-term incumbent", actionZh: "第九个方向把完整七任务均分提高到 47.13；增加第十项反而下降，因此回退到九项版本。", actionEn: "A ninth direction raises the complete seven-task mean to 47.13; a tenth term reduces the score, so the run returns to the nine-term checkpoint.", policy: "accept on improvement · rollback on decline", code: "final = center_9 if score(center_9) > score(center_10) else center_10", score: "47.13", nextZh: "重新加载九项 checkpoint 并保留为本次最好点。", nextEn: "Reload the nine-term checkpoint and retain it as the run best." },
    ],
  },
  minimax: {
    label: "MiniMax M2.7 · OpenCode",
    color: "#d85a32",
    target: "Qwen3-4B-Base",
    outcomeZh: "41.08 → 48.70（代表轨迹）",
    outcomeEn: "41.08 → 48.70 (representative trace)",
    summaryZh: "少量宽搜后迅速切换为成对 ES；它的关键收益来自把正反分数差转换为有状态的中心更新。",
    summaryEn: "After a small broad sweep, it quickly switched to paired ES; its main gain came from turning score differences into a stateful center update.",
    conclusionZh: "它用较少阶段把随机探测转换成连续中心更新；两次 Qwen3-4B 运行都接近 49，是当前样本中相对稳定的案例，但重复数仍然有限。",
    conclusionEn: "It turns random probes into consecutive center updates in relatively few stages. Both Qwen3-4B repeats are near 49, making this a comparatively stable case in the current sample, although the number of repeats remains small.",
    stages: [
      { titleZh: "宽尺度抽样", titleEn: "Broad scale sweep", actionZh: "围绕基础模型测试 8 个候选，尺度覆盖 0.0005–0.005。", actionEn: "Test eight candidates around the base across scales from 0.0005 to 0.005.", policy: "8 independent candidates · full suite", code: "scales = [.0005, .001, .002, .003, .005]", score: "44.38", nextZh: "用成对方向估计更可靠的更新。", nextEn: "Use paired directions for a more informative update." },
      { titleZh: "第一次 ES 更新", titleEn: "First ES update", actionZh: "评测 4 组正反方向，用分数差计算系数并合成两项中心。", actionEn: "Evaluate four antithetic pairs, turn score differences into coefficients, and build a two-term center.", policy: "4 pairs · σ=0.0005 · stateful", code: "coef = lr * (score_plus - score_minus)\ncenter += coef * direction", score: "48.54", nextZh: "在新中心附近再做一轮，而不是回到基础模型。", nextEn: "Probe around the new center instead of returning to the base." },
      { titleZh: "第二次 ES 更新", titleEn: "Second ES update", actionZh: "复用当前中心继续成对探测，得到最终的小幅提升。", actionEn: "Continue antithetic probing from the incumbent center for a final incremental gain.", policy: "centered pairs · full suite", code: "next_center = update(current_center, paired_scores)", score: "48.70", nextZh: "时间结束时由控制器保留最高完整分。", nextEn: "At timeout, the controller keeps the best complete score." },
    ],
  },
  glm: {
    label: "GLM-5.1 · OpenCode",
    color: "#3e9b70",
    target: "Qwen3-4B-Base",
    outcomeZh: "41.08 → 46.67（代表轨迹）",
    outcomeEn: "41.08 → 46.67 (representative trace)",
    summaryZh: "先做大范围随机尺度试验，排除破坏性步长；随后围绕单个有效 seed 做尺度细化，再用 ES 修正。",
    summaryEn: "It began with a wide random scale sweep to reject destructive steps, refined one useful seed, and then used ES-style corrections.",
    conclusionZh: "GLM 的轨迹较容易解释：先排除危险尺度，再复用一个有效方向，最后只接受一项修正。它的搜索上限不领先，但避免了无收益地持续增加组合项。",
    conclusionEn: "GLM's trace is comparatively easy to interpret: reject unsafe scales, reuse one useful direction, and accept only one corrective term. Its ceiling is not leading, but it avoids continuing to add unproductive components.",
    stages: [
      { titleZh: "大范围随机试验", titleEn: "Broad random sweep", actionZh: "测试 48 个候选，覆盖正负 0.0002–0.01；大尺度普遍破坏性能。", actionEn: "Test 48 candidates across positive and negative scales from 0.0002 to 0.01; large steps are usually destructive.", policy: "48 candidates · mixed scales", code: "for seed, scale in broad_grid:\n  evaluate(seed, scale)", score: "44.51", nextZh: "围绕最好 seed 缩小范围。", nextEn: "Narrow the search around the best seed." },
      { titleZh: "单方向尺度细化", titleEn: "Single-direction refinement", actionZh: "固定较好的方向，做更密的尺度扫描并找到约 0.00095 的局部点。", actionEn: "Fix a useful direction and run a denser scale sweep, locating a local point near 0.00095.", policy: "fixed seed · fine scale grid", code: "candidate(seed=best_seed, scale=.00095)", score: "45.24", nextZh: "尝试用分数差加入修正方向。", nextEn: "Use score differences to add a corrective direction." },
      { titleZh: "ES 修正与组合", titleEn: "ES correction and composition", actionZh: "加入第二个方向后升至 46.67；继续加入第三、第四项没有再提高。", actionEn: "Adding a second direction reaches 46.67; third and fourth terms do not improve it further.", policy: "incumbent + corrective terms", code: "best = base + term_1 + correction_2", score: "46.67", nextZh: "保留两项组合，停止无收益扩展。", nextEn: "Keep the two-term candidate and stop unproductive expansion." },
    ],
  },
};

export default function AgentTraceExplorer({ language }: { language: Language }) {
  const [agent, setAgent] = useState("minimax");
  const [stageIndex, setStageIndex] = useState(0);
  const trace = traces[agent];
  const stage = trace.stages[stageIndex];
  const tx = (zh: string, en: string) => language === "zh" ? zh : en;

  const selectAgent = (id: string) => { setAgent(id); setStageIndex(0); };
  return <figure className="agent-trace-explorer">
    <figcaption><b>{tx("交互 trace 阅读器。", "Interactive trace reader.")}</b> {tx("选择 Agent，再逐步查看它如何写搜索逻辑、改变采样策略，以及观察到分数后采取了什么行动。代码是根据完整 trace 还原的结构，不是逐字转录。", "Choose an agent and step through how it wrote search logic, changed its sampling policy, and what it did after observing a score. Code is a trace-derived structural sketch, not a verbatim transcript.")}</figcaption>
    <div className="trace-agent-tabs" role="tablist">
      {Object.entries(traces).map(([id, item]) => <button key={id} role="tab" aria-selected={agent === id} className={agent === id ? "active" : ""} style={{ "--trace-color": item.color } as CSSProperties} onClick={() => selectAgent(id)}>{item.label}</button>)}
    </div>
    <div className="trace-context"><div><span>{tx("目标模型", "Target model")}</span><b>{trace.target}</b></div><div><span>{tx("该案例结果", "Case outcome")}</span><b>{language === "zh" ? trace.outcomeZh : trace.outcomeEn}</b></div></div>
    <p className="trace-summary">{language === "zh" ? trace.summaryZh : trace.summaryEn}</p>
    <div className="trace-steps" style={{ "--trace-color": trace.color, "--trace-columns": trace.stages.length } as CSSProperties}>
      {trace.stages.map((item, index) => <button key={item.titleEn} className={stageIndex === index ? "active" : ""} onClick={() => setStageIndex(index)}><i>{index + 1}</i><span>{language === "zh" ? item.titleZh : item.titleEn}</span><small>{language === "zh" ? item.actionZh : item.actionEn}</small><b>{item.score}</b></button>)}
    </div>
    <div className="trace-detail">
      <div className="trace-decision"><span>{tx("研究决策", "Research decision")}</span><h4>{language === "zh" ? stage.titleZh : stage.titleEn}</h4><p>{language === "zh" ? stage.actionZh : stage.actionEn}</p><dl><div><dt>{tx("采样策略", "Sampling policy")}</dt><dd>{stage.policy}</dd></div><div><dt>{tx("当时最好分", "Running best")}</dt><dd>{stage.score}</dd></div></dl></div>
      <pre className="trace-code"><code>{stage.code}</code></pre>
      <div className="trace-next"><span>{tx("反馈后观察到的下一步", "Next step observed after feedback")}</span><p>{language === "zh" ? stage.nextZh : stage.nextEn}</p></div>
    </div>
    <div className="trace-conclusion" style={{ "--trace-color": trace.color } as CSSProperties}><b>{tx("案例结论", "Case conclusion")}</b><p>{language === "zh" ? trace.conclusionZh : trace.conclusionEn}</p></div>
  </figure>;
}
