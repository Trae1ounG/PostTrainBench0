import { readFile, writeFile } from "node:fs/promises";

const inventory = new URL(
  "../../zerogradbench/docs/data/qwen25_3b_inventory_supplemented_20260826_1922/",
  import.meta.url,
);
const output = new URL("../public/trajectory-data.json", import.meta.url);

function rows(text) {
  const [header, ...lines] = text.trim().split(/\r?\n/);
  const keys = header.split(",");
  return lines.map((line) => {
    const values = line.split(",");
    return Object.fromEntries(keys.map((key, index) => [key, values[index]]));
  });
}

const [trajectoryText, runText] = await Promise.all([
  readFile(new URL("all_run_trajectory.csv", inventory), "utf8"),
  readFile(new URL("all_runs.csv", inventory), "utf8"),
]);

const runMetadata = new Map(rows(runText).map((run) => [run.run_id, run]));
const grouped = new Map();

for (const row of rows(trajectoryText)) {
  const points = grouped.get(row.run_id) ?? [];
  points.push({
    minute: Number(row.elapsed_minutes),
    evaluation: Number(row.full_evaluation_index),
    score: Number(row.score),
    best: Number(row.running_best),
  });
  grouped.set(row.run_id, points);
}

const runs = [...grouped.entries()].map(([runId, points]) => {
  const meta = runMetadata.get(runId);
  const stride = Math.max(1, Math.ceil(points.length / 120));
  const kept = points.filter((point, index) => {
    if (index === 0 || index === points.length - 1 || index % stride === 0) return true;
    return point.best > points[index - 1].best;
  });

  return {
    runId,
    label: meta?.label ?? runId,
    agent: meta?.agent_model ?? "unknown",
    harness: meta?.harness ?? "unknown",
    kind: meta?.harness?.startsWith("fixed-baseline") ? "baseline" : "agent",
    accepted: meta?.formal_status === "valid",
    finalScore: Number(meta?.final_average_score || 0) || null,
    observedBest: Number(meta?.best_observed_full_suite_score || 0) || null,
    evaluations: Number(meta?.successful_full_suite_attempts || points.length),
    points: kept,
  };
});

runs.sort((a, b) => a.kind.localeCompare(b.kind) || a.agent.localeCompare(b.agent));
await writeFile(output, JSON.stringify({ generatedFrom: "all_run_trajectory.csv", runs }));
console.log(`Wrote ${runs.length} runs and ${runs.reduce((sum, run) => sum + run.points.length, 0)} plotted points.`);
