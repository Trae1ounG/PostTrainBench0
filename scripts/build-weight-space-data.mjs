import { readFile, writeFile } from "node:fs/promises";

const source = new URL(
  "../../runs/parameter_landscape/plots/qwen25_sigma0005_1000_landscape_20260811_1810/parameter_coordinates_and_scores.csv",
  import.meta.url,
);
const output = new URL("../public/weight-space-data.json", import.meta.url);
const tasks = ["countdown", "gsm8k", "math500", "olympiadbench", "mbpp", "rocstories", "uspto50k"];

const [header, ...lines] = (await readFile(source, "utf8")).trim().split(/\r?\n/);
const keys = header.split(",");
const points = lines.map((line) => {
  const row = Object.fromEntries(keys.map((key, index) => [key, line.split(",")[index]]));
  const scores = Object.fromEntries(tasks.map((task) => [task, Number(row[task])]));
  scores.joint = tasks.reduce((sum, task) => sum + scores[task], 0) / tasks.length;
  return {
    index: Number(row.candidate_index),
    seed: Number(row.seed),
    sigma: Number(row.sigma),
    pca: [Number(row.parameter_pca_x), Number(row.parameter_pca_y)],
    random: [Number(row.random_parameter_x), Number(row.random_parameter_y)],
    scores,
  };
});

await writeFile(output, JSON.stringify({ model: "Qwen2.5-3B-Instruct", candidates: points.length, tasks, points }));
console.log(`Wrote ${points.length} weight-space points.`);
