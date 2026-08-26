"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Point = { minute: number; evaluation: number; score: number; best: number };
type Run = {
  runId: string;
  label: string;
  agent: string;
  harness: string;
  kind: "agent" | "baseline";
  accepted: boolean;
  finalScore: number | null;
  observedBest: number | null;
  evaluations: number;
  points: Point[];
};
type Dataset = { runs: Run[] };
type Language = "zh" | "en";
type XAxis = "minute" | "evaluation";
type ViewMode = "summary" | "runs";
type ChartPoint = {
  x: number;
  best: number;
  low: number;
  high: number;
  source?: Point;
  milestone?: boolean;
  descriptionZh?: string;
  descriptionEn?: string;
};
type Series = {
  id: string;
  agent: string;
  label: string;
  harness: string;
  accepted: boolean;
  kind: "agent" | "baseline";
  runCount: number;
  color: string;
  points: ChartPoint[];
};
type Hover = { series: Series; point: ChartPoint; px: number; py: number };

const BASE_SCORE = 0.440764;
const TIME_LIMIT_MINUTES = 240;
const displayScore = (value: number) => (value * 100).toFixed(2);
const COLORS = ["#2457ff", "#de5b3f", "#078b71", "#8a4bd0", "#d39400", "#1686b0", "#cc3d7e", "#667085", "#76a000", "#6b4f3f", "#111827"];

function cleanAgentName(agent: string, language: Language) {
  const [model, effort] = agent.split(":");
  const suffix = effort ? ` ${effort}` : "";
  if (agent === "es-conservative") return language === "zh" ? "进化策略 · 保守更新" : "Evolution strategy · conservative";
  if (agent === "es-original") return language === "zh" ? "进化策略 · 原始更新" : "Evolution strategy · original";
  return model
    .replace("claude-4.6-sonnet-medium", "Sonnet 4.6 medium")
    .replace("claude-opus-4-8-high", "Opus 4.8 high")
    .replace("ali-deepseek-v4-pro", "DeepSeek V4 Pro")
    .replace("gpt-5.4-pro-2026-03-05", "GPT-5.4 Pro")
    .replace("gpt-5.6-sol", "GPT-5.6")
    .replace("kimi-k2.6", "Kimi K2.6")
    .replace("glm-5.1", "GLM-5.1")
    .replace("openai_qwen3.7-max", "Qwen3.7-Max")
    .replace("Minimax-M2.7-highspeed", "MiniMax M2.7")
    .replace("randopt", "RandOPT") + suffix;
}

function agentGroup(run: Run) {
  if (run.agent === "es") return run.runId.includes("conservative") ? "es-conservative" : "es-original";
  const effort = run.label.match(/\[(medium|high|xhigh)\]/)?.[1];
  if (effort) return `${run.agent}:${effort}`;
  return run.agent;
}

function isExcludedRun(run: Run) {
  return run.agent === "gpt-5.5-2026-04-24";
}

function scoreOf(run: Run) {
  return run.observedBest ?? run.finalScore;
}

function topTwoPerSetting(runs: Run[]) {
  const grouped = new Map<string, Run[]>();
  for (const run of runs) {
    const key = `${agentGroup(run)}|${run.harness}`;
    grouped.set(key, [...(grouped.get(key) ?? []), run]);
  }
  return [...grouped.values()].flatMap((group) => [...group]
    .filter((run) => scoreOf(run) !== null)
    .sort((left, right) => (scoreOf(right) ?? -Infinity) - (scoreOf(left) ?? -Infinity))
    .slice(0, 2));
}

function median(values: number[]) {
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[middle] : (sorted[middle - 1] + sorted[middle]) / 2;
}

function metric(values: number[], digits = 1) {
  if (!values.length) return "—";
  const center = median(values);
  const low = Math.min(...values);
  const high = Math.max(...values);
  return low === high ? center.toFixed(digits) : `${center.toFixed(digits)} (${low.toFixed(digits)}–${high.toFixed(digits)})`;
}

function valueAt(run: Run, x: number, axis: XAxis) {
  let value = BASE_SCORE;
  for (const point of run.points) {
    const pointX = axis === "minute" ? point.minute : point.evaluation;
    if (pointX > x) break;
    value = point.best;
  }
  return value;
}

function pointsForRun(run: Run, axis: XAxis): ChartPoint[] {
  let previousBest = BASE_SCORE;
  const points = run.points.map((point, index) => {
    const improved = point.best > previousBest + 1e-9;
    const prior = previousBest;
    previousBest = Math.max(previousBest, point.best);
    const x = axis === "minute" ? Math.min(point.minute, TIME_LIMIT_MINUTES) : point.evaluation;
    return {
      x,
      best: point.best,
      low: point.best,
      high: point.best,
      source: point,
      milestone: improved || index === 0,
      descriptionZh: improved
        ? `第 ${point.evaluation} 次完整评测刷新最好分：${displayScore(prior)} → ${displayScore(point.best)}。`
        : `第 ${point.evaluation} 次完整评测没有刷新最好分，当前最好分保持 ${displayScore(point.best)}。`,
      descriptionEn: improved
        ? `Full evaluation ${point.evaluation} raises the incumbent from ${displayScore(prior)} to ${displayScore(point.best)}.`
        : `Full evaluation ${point.evaluation} leaves the incumbent at ${displayScore(point.best)}.`,
    };
  });
  if (axis === "minute" && points.length) {
    const endpoint = points.at(-1)!;
    if (endpoint.x < TIME_LIMIT_MINUTES) {
      points.push({
        x: TIME_LIMIT_MINUTES,
        best: endpoint.best,
        low: endpoint.low,
        high: endpoint.high,
        milestone: false,
        descriptionZh: `按四小时预算展示，最后观测到的最好分 ${displayScore(endpoint.best)} 保持到第 240 分钟。`,
        descriptionEn: `Shown over the full four-hour budget; the last observed incumbent, ${displayScore(endpoint.best)}, is carried to minute 240.`,
      });
    }
  }
  return points;
}

export default function InteractiveTrajectory({ language }: { language: Language }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<Dataset | null>(null);
  const [width, setWidth] = useState(850);
  const [xAxis, setXAxis] = useState<XAxis>("minute");
  const [viewMode, setViewMode] = useState<ViewMode>("summary");
  const [showBaselines, setShowBaselines] = useState(false);
  const [enabledAgents, setEnabledAgents] = useState<Set<string> | null>(null);
  const [selectedAgent, setSelectedAgent] = useState("gpt-5.6-sol");
  const [hover, setHover] = useState<Hover | null>(null);
  const tr = (zh: string, en: string) => language === "zh" ? zh : en;

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}trajectory-data.json`).then((response) => response.json()).then(setData);
  }, []);

  useEffect(() => {
    if (!wrapRef.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(300, Math.floor(entry.contentRect.width))));
    observer.observe(wrapRef.current);
    return () => observer.disconnect();
  }, []);

  const agents = useMemo(() => {
    if (!data) return [];
    return [...new Set(data.runs.filter((run) => !run.runId.includes("smoke") && !isExcludedRun(run)).map(agentGroup))].sort();
  }, [data]);

  const colorByAgent = useMemo(
    () => new Map(agents.map((agent, index) => [agent, COLORS[index % COLORS.length]])),
    [agents],
  );

  const eligibleRuns = useMemo(() => {
    if (!data) return [];
    return topTwoPerSetting(data.runs.filter((run) =>
      !run.runId.includes("smoke") &&
      !isExcludedRun(run) &&
      (showBaselines || run.kind === "agent"),
    ));
  }, [data, showBaselines]);

  const displayedAgents = useMemo(
    () => agents.filter((agent) => showBaselines || data?.runs.find((run) => agentGroup(run) === agent)?.kind === "agent"),
    [agents, data, showBaselines],
  );
  const activeSelectedAgent = displayedAgents.includes(selectedAgent) ? selectedAgent : displayedAgents[0];

  const series = useMemo<Series[]>(() => {
    if (!eligibleRuns.length) return [];
    if (viewMode === "runs") {
      return eligibleRuns
        .filter((run) => agentGroup(run) === activeSelectedAgent)
        .map((run, index) => ({
          id: run.runId,
          agent: agentGroup(run),
          label: `${cleanAgentName(agentGroup(run), language)} · ${language === "zh" ? "运行" : "run"} ${index + 1}`,
          harness: run.harness,
          accepted: run.accepted,
          kind: run.kind,
          runCount: 1,
          color: colorByAgent.get(agentGroup(run)) ?? "#2457ff",
          points: pointsForRun(run, xAxis),
        }));
    }

    const maxX = xAxis === "minute"
      ? TIME_LIMIT_MINUTES
      : Math.max(1, ...eligibleRuns.flatMap((run) => run.points.map((point) => point.evaluation)));
    return displayedAgents
      .filter((agent) => !enabledAgents || enabledAgents.has(agent))
      .map((agent) => {
        const runs = eligibleRuns.filter((run) => agentGroup(run) === agent);
        if (!runs.length) return null;
        const points = Array.from({ length: 121 }, (_, index) => {
          const x = (maxX * index) / 120;
          const values = runs.map((run) => valueAt(run, x, xAxis));
          return { x, best: median(values), low: Math.min(...values), high: Math.max(...values) };
        });
        return {
          id: agent,
          agent,
          label: cleanAgentName(agent, language),
          harness: [...new Set(runs.map((run) => run.harness))].join(" / "),
          accepted: runs.every((run) => run.accepted),
          kind: runs[0].kind,
          runCount: runs.length,
          color: colorByAgent.get(agent) ?? "#2457ff",
          points,
        };
      })
      .filter((item): item is Series => item !== null);
  }, [activeSelectedAgent, colorByAgent, displayedAgents, eligibleRuns, enabledAgents, language, viewMode, xAxis]);

  const geometry = useMemo(() => {
    const height = width < 620 ? 390 : 450;
    const margin = width < 620 ? { left: 48, right: 14, top: 24, bottom: 48 } : { left: 64, right: 25, top: 28, bottom: 54 };
    const maxX = xAxis === "minute"
      ? TIME_LIMIT_MINUTES
      : Math.max(1, ...series.flatMap((item) => item.points.map((point) => point.x)));
    const allY = series.flatMap((item) => item.points.flatMap((point) => [point.low, point.high]));
    const rawMin = Math.min(BASE_SCORE, ...allY);
    const rawMax = Math.max(BASE_SCORE, ...allY);
    const padding = Math.max(0.004, (rawMax - rawMin) * 0.08);
    const minY = Math.floor((rawMin - padding) * 200) / 200;
    const maxY = Math.ceil((rawMax + padding) * 200) / 200;
    return {
      height, margin, maxX, minY, maxY,
      x: (value: number) => margin.left + (value / maxX) * (width - margin.left - margin.right),
      y: (value: number) => margin.top + ((maxY - value) / (maxY - minY)) * (height - margin.top - margin.bottom),
    };
  }, [series, width, xAxis]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio;
    canvas.height = geometry.height * ratio;
    canvas.style.width = `${width}px`;
    canvas.style.height = `${geometry.height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.fillStyle = "#ffffff";
    context.fillRect(0, 0, width, geometry.height);

    context.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.textAlign = "right";
    context.textBaseline = "middle";
    for (let i = 0; i <= 5; i++) {
      const value = geometry.minY + ((geometry.maxY - geometry.minY) * i) / 5;
      const y = geometry.y(value);
      context.strokeStyle = "#e7e8e4";
      context.lineWidth = 1;
      context.beginPath(); context.moveTo(geometry.margin.left, y); context.lineTo(width - geometry.margin.right, y); context.stroke();
      context.fillStyle = "#7b8190";
      context.fillText(displayScore(value), geometry.margin.left - 9, y);
    }

    context.textAlign = "center";
    context.textBaseline = "top";
    for (let i = 0; i <= 5; i++) {
      const value = (geometry.maxX * i) / 5;
      const x = geometry.x(value);
      context.strokeStyle = "#f0f0ec";
      context.beginPath(); context.moveTo(x, geometry.margin.top); context.lineTo(x, geometry.height - geometry.margin.bottom); context.stroke();
      context.fillStyle = "#7b8190";
      context.fillText(xAxis === "minute" ? `${Math.round(value)}${language === "zh" ? " 分" : "m"}` : `${Math.round(value)}${language === "zh" ? " 次" : ""}`, x, geometry.height - geometry.margin.bottom + 12);
    }

    const baseY = geometry.y(BASE_SCORE);
    context.save();
    context.setLineDash([5, 5]); context.strokeStyle = "#545c6a"; context.lineWidth = 1;
    context.beginPath(); context.moveTo(geometry.margin.left, baseY); context.lineTo(width - geometry.margin.right, baseY); context.stroke();
    context.restore();

    if (viewMode === "summary") {
      for (const item of series) {
        if (item.runCount < 2) continue;
        context.save();
        context.fillStyle = item.color;
        context.globalAlpha = 0.10;
        context.beginPath();
        item.points.forEach((point, index) => {
          const x = geometry.x(point.x); const y = geometry.y(point.high);
          if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
        });
        [...item.points].reverse().forEach((point) => context.lineTo(geometry.x(point.x), geometry.y(point.low)));
        context.closePath(); context.fill(); context.restore();
      }
    }

    for (const item of series) {
      context.save();
      context.strokeStyle = item.color;
      context.globalAlpha = hover && hover.series.id !== item.id ? 0.18 : viewMode === "runs" ? 0.62 : 0.9;
      context.lineWidth = hover?.series.id === item.id ? 3.4 : viewMode === "summary" ? 2.4 : 1.6;
      context.beginPath();
      item.points.forEach((point, index) => {
        const x = geometry.x(point.x); const y = geometry.y(point.best);
        if (index === 0) context.moveTo(x, y); else context.lineTo(x, y);
      });
      context.stroke(); context.restore();
    }

    if (viewMode === "runs") {
      for (const item of series) {
        for (const point of item.points) {
          if (!point.milestone) continue;
          context.save();
          context.fillStyle = "#ffffff";
          context.strokeStyle = item.color;
          context.lineWidth = 1.8;
          context.beginPath();
          context.arc(geometry.x(point.x), geometry.y(point.best), 3.4, 0, Math.PI * 2);
          context.fill();
          context.stroke();
          context.restore();
        }
      }
    }

    if (hover) {
      context.save();
      context.strokeStyle = "#7b8190"; context.setLineDash([3, 4]);
      context.beginPath(); context.moveTo(hover.px, geometry.margin.top); context.lineTo(hover.px, geometry.height - geometry.margin.bottom); context.stroke();
      context.fillStyle = hover.series.color;
      context.beginPath(); context.arc(hover.px, hover.py, 5, 0, Math.PI * 2); context.fill();
      context.strokeStyle = "white"; context.lineWidth = 2; context.stroke(); context.restore();
    }
  }, [geometry, hover, language, series, viewMode, width, xAxis]);

  function pointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    const rect = event.currentTarget.getBoundingClientRect();
    const px = event.clientX - rect.left;
    const py = event.clientY - rect.top;
    let nearest: (Hover & { distance: number }) | null = null;
    for (const item of series) {
      for (const point of item.points) {
        const pointX = geometry.x(point.x); const pointY = geometry.y(point.best);
        const distance = Math.hypot(pointX - px, (pointY - py) * 0.38);
        if (!nearest || distance < nearest.distance) nearest = { series: item, point, px: pointX, py: pointY, distance };
      }
    }
    setHover(nearest && nearest.distance < 40 ? nearest : null);
  }

  function toggleAgent(agent: string) {
    if (viewMode === "runs") {
      setSelectedAgent(agent); setHover(null); return;
    }
    setEnabledAgents((current) => {
      const next = new Set(current ?? displayedAgents);
      if (next.has(agent) && next.size > 1) next.delete(agent); else next.add(agent);
      return next;
    });
    setHover(null);
  }

  const selectedRuns = eligibleRuns.filter((run) => agentGroup(run) === activeSelectedAgent);
  const selectedEnds = selectedRuns.map((run) => run.points.at(-1)?.best ?? BASE_SCORE);
  const readoutSeries = hover?.series;
  const trajectoryStats = displayedAgents.map((agent) => {
    const runs = eligibleRuns.filter((run) => agentGroup(run) === agent);
    const firstImprovements = runs.flatMap((run) => {
      const point = run.points.find((candidate) => candidate.best > BASE_SCORE + 1e-9);
      return point ? [point.minute] : [];
    });
    const bestTimes = runs.flatMap((run) => {
      const endpoint = run.points.at(-1)?.best;
      const point = endpoint === undefined ? undefined : run.points.find((candidate) => candidate.best >= endpoint - 1e-9);
      return point ? [point.minute] : [];
    });
    const improvementCounts = runs.map((run) => {
      let incumbent = BASE_SCORE;
      let count = 0;
      for (const point of run.points) {
        if (point.best > incumbent + 1e-9) { incumbent = point.best; count += 1; }
      }
      return count;
    });
    const evaluations = runs.map((run) => run.evaluations);
    const endpoints = runs.map((run) => (run.points.at(-1)?.best ?? BASE_SCORE) * 100);
    const gainsAfterTwoHours = runs.map((run) => ((run.points.at(-1)?.best ?? BASE_SCORE) - valueAt(run, 120, "minute")) * 100);
    return { agent, runs, firstImprovements, bestTimes, improvementCounts, evaluations, endpoints, gainsAfterTwoHours };
  }).filter((row) => row.runs.length);

  if (!data) return <div className="trajectory-loading">{tr("正在加载实验轨迹…", "Loading search trajectories…")}</div>;

  return (
    <div className="trajectory-explorer">
      <div className="trajectory-controls">
        <div className="control-group">
          <span>{tr("显示方式", "View")}</span>
          <div className="segmented small">
            <button className={viewMode === "summary" ? "active" : ""} onClick={() => { setViewMode("summary"); setHover(null); }}>{tr("按 Agent 汇总", "Agent summary")}</button>
            <button className={viewMode === "runs" ? "active" : ""} onClick={() => { setViewMode("runs"); setHover(null); }}>{tr("查看单次运行", "Individual runs")}</button>
          </div>
        </div>
        <div className="control-group">
          <span>{tr("横轴", "X axis")}</span>
          <div className="segmented small">
            <button className={xAxis === "minute" ? "active" : ""} onClick={() => setXAxis("minute")}>{tr("运行时间", "Elapsed time")}</button>
            <button className={xAxis === "evaluation" ? "active" : ""} onClick={() => setXAxis("evaluation")}>{tr("完整评测次数", "Full evaluations")}</button>
          </div>
        </div>
        <label className="switch-control"><input type="checkbox" checked={showBaselines} onChange={(event) => { setShowBaselines(event.target.checked); setHover(null); }} /><span />{tr("显示固定搜索基线", "Show fixed baselines")}</label>
      </div>

      <div className="trajectory-legend" aria-label={viewMode === "summary" ? tr("切换 Agent 汇总曲线", "Toggle agent summary curves") : tr("选择一个 Agent", "Select an agent")}>
        {displayedAgents.map((agent) => {
          const runCount = eligibleRuns.filter((run) => agentGroup(run) === agent).length;
          const enabled = viewMode === "runs" ? agent === activeSelectedAgent : !enabledAgents || enabledAgents.has(agent);
          return (
            <button key={agent} className={enabled ? "enabled" : ""} onClick={() => toggleAgent(agent)} style={{ "--series-color": colorByAgent.get(agent) } as React.CSSProperties}>
              <i />{cleanAgentName(agent, language)} <small>{runCount}</small>
            </button>
          );
        })}
      </div>

      <div className="trajectory-stage">
        <div className="trajectory-canvas-wrap" ref={wrapRef}>
          <span className="y-axis-title">{tr("当前最佳七任务均分（0–100）", "Best seven-task mean so far (0–100)")}</span>
          <canvas ref={canvasRef} onPointerMove={pointerMove} onPointerLeave={() => setHover(null)} aria-label={tr("交互式最佳分数搜索轨迹", "Interactive best-score search trajectories")} />
        </div>
        <aside className="trajectory-readout">
          <span className="readout-kicker">{hover ? tr("当前指向的数据", "Point under cursor") : viewMode === "summary" ? tr("如何阅读", "How to read") : tr("当前 Agent", "Selected Agent")}</span>
          {hover ? (
            <>
              <strong>{readoutSeries?.label}</strong>
              <p>{readoutSeries?.harness}</p>
              {viewMode === "runs" && hover.point.descriptionZh && <p className="trajectory-point-description">{language === "zh" ? hover.point.descriptionZh : hover.point.descriptionEn}</p>}
              <dl>
                <div><dt>{viewMode === "summary" ? tr("中位最佳分", "Median best") : tr("最佳分数", "Best score")}</dt><dd>{displayScore(hover.point.best)}</dd></div>
                {viewMode === "summary" && <div><dt>{tr("重复运行范围", "Run range")}</dt><dd>{displayScore(hover.point.low)}–{displayScore(hover.point.high)}</dd></div>}
                <div><dt>{xAxis === "minute" ? tr("已运行", "Elapsed") : tr("完整评测", "Full evaluation")}</dt><dd>{xAxis === "minute" ? `${hover.point.x.toFixed(1)} ${tr("分钟", "min")}` : `#${Math.round(hover.point.x)}`}</dd></div>
                <div><dt>{tr("计入运行", "Runs represented")}</dt><dd>{hover.series.runCount}</dd></div>
              </dl>
            </>
          ) : viewMode === "summary" ? (
            <>
              <strong>{tr("每条实线代表一种 Agent 设置", "One line per Agent setting")}</strong>
              <p>{tr("实线表示同一设置最高两次完整运行的当前最佳分数中位数，阴影覆盖两次运行的范围。没有最终提交但产生完整七任务分数的运行，按其观测到的最高完整分计入。", "The line is the median best-so-far score across the two highest-scoring complete runs for a setting, and the band spans those runs. A run without a final submission is still included by its best observed complete seven-task score.")}</p>
              <div className="readout-swatch"><i /> {tr("中位轨迹", "median")} <span /> {tr("运行范围", "run range")}</div>
            </>
          ) : (
            <>
              <strong>{cleanAgentName(activeSelectedAgent ?? "", language)}</strong>
              <p>{tr("每条线表示一次得到完整七任务分数的搜索运行。", "Each line is one search run that produced a complete seven-task score.")}</p>
              <dl>
                <div><dt>{tr("可见运行", "Visible runs")}</dt><dd>{selectedRuns.length}</dd></div>
                <div><dt>{tr("终点中位数", "Endpoint median")}</dt><dd>{selectedEnds.length ? displayScore(median(selectedEnds)) : "—"}</dd></div>
                <div><dt>{tr("终点范围", "Endpoint range")}</dt><dd>{selectedEnds.length ? `${displayScore(Math.min(...selectedEnds))}–${displayScore(Math.max(...selectedEnds))}` : "—"}</dd></div>
              </dl>
            </>
          )}
        </aside>
      </div>

      <div className="trajectory-footer">
        <span>{tr("当前显示", "Showing")} <b>{series.length}</b> {viewMode === "summary" ? tr("条 Agent 汇总曲线", "Agent summaries") : tr("条单次运行曲线", "individual runs")}</span>
        <span>{tr("每个精确设置最多保留最高两次", "At most the top two runs per exact setting")}</span>
        <span>{tr("水平虚线：未修改模型", "Dashed horizontal line: unmodified model")} {displayScore(BASE_SCORE)}</span>
        <span>{xAxis === "minute" ? tr("时间包含评测等待", "Time includes evaluation latency") : tr("一次完整评测覆盖全部七项任务", "One full evaluation covers all seven tasks")}</span>
      </div>
      <div className="trajectory-detail-table">
        <div className="explorer-head"><div><strong>{tr("关键时间点与搜索效率", "Milestones and search efficiency")}</strong><span>{tr("数值为所保留运行的中位数；括号内为最低–最高。时间从 Agent 启动开始计算。", "Values are medians across retained runs; parentheses show min–max. Time starts when the Agent starts.")}</span></div></div>
        <div className="table-scroll"><table className="data-table"><thead><tr>
          <th>{tr("Agent 设置", "Agent setting")}</th><th>n</th>
          <th>{tr("首次超过基础分（分钟）", "First gain (min)")}</th>
          <th>{tr("达到最好分（分钟）", "Best reached (min)")}</th>
          <th>{tr("刷新最好分次数", "Incumbent updates")}</th>
          <th>{tr("完整评测数", "Full evaluations")}</th>
          <th>{tr("两小时后新增分数", "Gain after 2h")}</th>
          <th>{tr("最好分数", "Best score")}</th>
        </tr></thead><tbody>{trajectoryStats.map((row) => <tr key={row.agent}>
          <td><b>{cleanAgentName(row.agent, language)}</b><small className="table-harness">{[...new Set(row.runs.map((run) => run.harness))].join(" / ")}</small></td>
          <td>{row.runs.length}</td><td>{metric(row.firstImprovements)}</td><td>{metric(row.bestTimes)}</td>
          <td>{metric(row.improvementCounts, 0)}</td><td>{metric(row.evaluations, 0)}</td>
          <td>{metric(row.gainsAfterTwoHours, 2)}</td><td>{metric(row.endpoints, 2)}</td>
        </tr>)}</tbody></table></div>
        <p className="figure-source">{tr("“两小时后新增分数” = 四小时内最好分 − 第 120 分钟时的 running-best；它区分早期发现与后半程继续优化。", "Gain after 2h = final running-best minus the running-best at minute 120; it separates early discovery from continued improvement in the second half.")}</p>
      </div>
    </div>
  );
}
