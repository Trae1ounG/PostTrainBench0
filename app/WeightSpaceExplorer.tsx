"use client";

import { useEffect, useMemo, useRef, useState } from "react";

type Language = "zh" | "en";
type Projection = "pca" | "random";
type ScoreKey = "joint" | "countdown" | "gsm8k" | "math500" | "olympiadbench" | "mbpp" | "rocstories" | "uspto50k";
type WeightPoint = { index: number; seed: number; sigma: number; pca: [number, number]; random: [number, number]; scores: Record<ScoreKey, number> };
type Dataset = { model: string; candidates: number; points: WeightPoint[] };

const scoreNames: Record<ScoreKey, string> = {
  joint: "七任务均分", countdown: "Countdown", gsm8k: "GSM8K", math500: "MATH-500",
  olympiadbench: "OlympiadBench", mbpp: "MBPP", rocstories: "ROCStories", uspto50k: "USPTO-50K",
};

function color(value: number) {
  const t = Math.max(0, Math.min(1, value));
  if (t < 0.5) {
    const p = t * 2;
    return `rgb(${Math.round(34 + 221 * p)},${Math.round(70 + 184 * p)},${Math.round(170 + 70 * p)})`;
  }
  const p = (t - 0.5) * 2;
  return `rgb(${Math.round(255 - 47 * p)},${Math.round(254 - 188 * p)},${Math.round(240 - 194 * p)})`;
}

export default function WeightSpaceExplorer({ language }: { language: Language }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);
  const [data, setData] = useState<Dataset | null>(null);
  const [width, setWidth] = useState(820);
  const [projection, setProjection] = useState<Projection>("pca");
  const [scoreKey, setScoreKey] = useState<ScoreKey>("joint");
  const [hover, setHover] = useState<{ point: WeightPoint; x: number; y: number } | null>(null);
  const tr = (zh: string, en: string) => language === "zh" ? zh : en;

  useEffect(() => {
    fetch(`${import.meta.env.BASE_URL}weight-space-data.json`).then((response) => response.json()).then(setData);
  }, []);

  useEffect(() => {
    if (!wrapRef.current) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(300, Math.floor(entry.contentRect.width))));
    observer.observe(wrapRef.current);
    return () => observer.disconnect();
  }, []);

  const plot = useMemo(() => {
    if (!data) return null;
    const points = data.points;
    const xs = points.map((point) => point[projection][0]);
    const ys = points.map((point) => point[projection][1]);
    const scores = points.map((point) => point.scores[scoreKey]);
    const minX = Math.min(...xs); const maxX = Math.max(...xs);
    const minY = Math.min(...ys); const maxY = Math.max(...ys);
    const sorted = [...scores].sort((a, b) => a - b);
    const low = sorted[Math.floor(sorted.length * 0.05)];
    const high = sorted[Math.floor(sorted.length * 0.95)];
    const best = points.reduce((winner, point) => point.scores[scoreKey] > winner.scores[scoreKey] ? point : winner, points[0]);
    return { points, minX, maxX, minY, maxY, low, high, best };
  }, [data, projection, scoreKey]);

  const geometry = useMemo(() => {
    const height = width < 620 ? 430 : 540;
    const margin = { left: 44, right: 18, top: 20, bottom: 42 };
    const x = (value: number) => margin.left + ((value - (plot?.minX ?? 0)) / Math.max(1e-9, (plot?.maxX ?? 1) - (plot?.minX ?? 0))) * (width - margin.left - margin.right);
    const y = (value: number) => margin.top + (((plot?.maxY ?? 1) - value) / Math.max(1e-9, (plot?.maxY ?? 1) - (plot?.minY ?? 0))) * (height - margin.top - margin.bottom);
    return { height, margin, x, y };
  }, [plot, width]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !plot) return;
    const ratio = window.devicePixelRatio || 1;
    canvas.width = width * ratio; canvas.height = geometry.height * ratio;
    canvas.style.width = `${width}px`; canvas.style.height = `${geometry.height}px`;
    const context = canvas.getContext("2d");
    if (!context) return;
    context.scale(ratio, ratio);
    context.fillStyle = "#f8f8f5"; context.fillRect(0, 0, width, geometry.height);

    const columns = width < 620 ? 35 : 56; const rows = width < 620 ? 34 : 42;
    const sums = Array.from({ length: columns * rows }, () => 0);
    const counts = Array.from({ length: columns * rows }, () => 0);
    for (const point of plot.points) {
      const [px, py] = point[projection];
      const column = Math.max(0, Math.min(columns - 1, Math.floor(((px - plot.minX) / (plot.maxX - plot.minX)) * columns)));
      const row = Math.max(0, Math.min(rows - 1, Math.floor(((plot.maxY - py) / (plot.maxY - plot.minY)) * rows)));
      const index = row * columns + column;
      sums[index] += point.scores[scoreKey]; counts[index] += 1;
    }
    for (let pass = 0; pass < 3; pass++) {
      const nextSums = [...sums]; const nextCounts = [...counts];
      for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
        const index = row * columns + column;
        if (counts[index]) continue;
        let sum = 0; let count = 0;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nr = row + dy; const nc = column + dx;
          if (nr < 0 || nr >= rows || nc < 0 || nc >= columns) continue;
          const neighbor = nr * columns + nc;
          if (counts[neighbor]) { sum += sums[neighbor] / counts[neighbor]; count += 1; }
        }
        if (count) { nextSums[index] = sum; nextCounts[index] = count; }
      }
      sums.splice(0, sums.length, ...nextSums); counts.splice(0, counts.length, ...nextCounts);
    }
    const cellWidth = (width - geometry.margin.left - geometry.margin.right) / columns;
    const cellHeight = (geometry.height - geometry.margin.top - geometry.margin.bottom) / rows;
    for (let row = 0; row < rows; row++) for (let column = 0; column < columns; column++) {
      const index = row * columns + column;
      if (!counts[index]) continue;
      const value = sums[index] / counts[index];
      context.fillStyle = color((value - plot.low) / Math.max(1e-9, plot.high - plot.low));
      context.globalAlpha = 0.72;
      context.fillRect(geometry.margin.left + column * cellWidth, geometry.margin.top + row * cellHeight, cellWidth + 1, cellHeight + 1);
    }
    context.globalAlpha = 1;

    for (const point of plot.points) {
      const [px, py] = point[projection];
      context.fillStyle = "rgba(17,24,39,.34)";
      context.beginPath(); context.arc(geometry.x(px), geometry.y(py), 1.8, 0, Math.PI * 2); context.fill();
    }

    const [bestX, bestY] = plot.best[projection];
    const bx = geometry.x(bestX); const by = geometry.y(bestY);
    context.fillStyle = "#dff56b"; context.strokeStyle = "#111827"; context.lineWidth = 1.5;
    context.beginPath();
    for (let i = 0; i < 10; i++) {
      const radius = i % 2 ? 4 : 9; const angle = -Math.PI / 2 + (i * Math.PI) / 5;
      const x = bx + Math.cos(angle) * radius; const y = by + Math.sin(angle) * radius;
      if (i === 0) context.moveTo(x, y); else context.lineTo(x, y);
    }
    context.closePath(); context.fill(); context.stroke();

    if (hover) {
      context.strokeStyle = "white"; context.lineWidth = 2.5;
      context.beginPath(); context.arc(hover.x, hover.y, 6, 0, Math.PI * 2); context.stroke();
      context.strokeStyle = "#111827"; context.lineWidth = 1;
      context.beginPath(); context.arc(hover.x, hover.y, 7.5, 0, Math.PI * 2); context.stroke();
    }
  }, [geometry, hover, plot, projection, scoreKey, width]);

  function pointerMove(event: React.PointerEvent<HTMLCanvasElement>) {
    if (!plot) return;
    const rect = event.currentTarget.getBoundingClientRect();
    const x = event.clientX - rect.left; const y = event.clientY - rect.top;
    let nearest: { point: WeightPoint; x: number; y: number; distance: number } | null = null;
    for (const point of plot.points) {
      const pointX = geometry.x(point[projection][0]); const pointY = geometry.y(point[projection][1]);
      const distance = Math.hypot(pointX - x, pointY - y);
      if (!nearest || distance < nearest.distance) nearest = { point, x: pointX, y: pointY, distance };
    }
    setHover(nearest && nearest.distance < 16 ? nearest : null);
  }

  if (!data || !plot) return <div className="weight-space-loading">{tr("正在加载参数空间…", "Loading weight space…")}</div>;
  const selected = hover?.point ?? plot.best;

  return (
    <div className="weight-space-explorer">
      <div className="weight-controls">
        <div className="control-group"><span>{tr("降维方式", "Projection")}</span><div className="segmented small">
          <button className={projection === "pca" ? "active" : ""} onClick={() => { setProjection("pca"); setHover(null); }}>PCA</button>
          <button className={projection === "random" ? "active" : ""} onClick={() => { setProjection("random"); setHover(null); }}>{tr("固定随机投影", "Fixed random projection")}</button>
        </div></div>
        <label className="score-select"><span>{tr("着色分数", "Color by")}</span><select value={scoreKey} onChange={(event) => { setScoreKey(event.target.value as ScoreKey); setHover(null); }}>
          {(Object.keys(scoreNames) as ScoreKey[]).map((key) => <option key={key} value={key}>{key === "joint" ? tr(scoreNames[key], "Seven-task mean") : scoreNames[key]}</option>)}
        </select></label>
      </div>
      <div className="weight-stage">
        <div className="weight-canvas" ref={wrapRef}><canvas ref={canvasRef} onPointerMove={pointerMove} onPointerLeave={() => setHover(null)} aria-label={tr("模型参数空间热力图", "Model weight-space heatmap")} /></div>
        <aside className="weight-readout">
          <span>{hover ? tr("当前候选", "Candidate under cursor") : tr("当前任务最高分候选", "Best candidate for selected score")}</span>
          <strong>#{selected.index}</strong>
          <dl>
            <div><dt>{tr("当前着色分数", "Selected score")}</dt><dd>{selected.scores[scoreKey].toFixed(4)}</dd></div>
            <div><dt>{tr("七任务均分", "Seven-task mean")}</dt><dd>{selected.scores.joint.toFixed(4)}</dd></div>
            <div><dt>{tr("扰动幅度", "Noise scale")}</dt><dd>{selected.sigma}</dd></div>
            <div><dt>{tr("方向种子", "Direction seed")}</dt><dd>{selected.seed}</dd></div>
          </dl>
          <p>{tr("黄色星号表示当前着色指标下实际测得的最高分候选；黑点是全部 1,000 个真实扰动。", "The yellow star is the measured best candidate for the selected score; black dots are all 1,000 real perturbations.")}</p>
        </aside>
      </div>
      <div className="weight-footer"><span>{tr("蓝色：低分区域", "Blue: lower scores")}</span><i /><span>{tr("红色：高分区域", "Red: higher scores")}</span><b>{tr("坐标只由权重差计算，不使用任务分数", "Coordinates use weight differences only, never task scores")}</b></div>
    </div>
  );
}
