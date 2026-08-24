import assert from "node:assert/strict";
import { access, readFile, readdir } from "node:fs/promises";
import test from "node:test";

const root = new URL("../", import.meta.url);

test("builds the bilingual research article for the personal site", async () => {
  const [html, page, main] = await Promise.all([
    readFile(new URL("dist-pages/index.html", root), "utf8"),
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("main.tsx", root), "utf8"),
  ]);

  assert.match(html, /<html lang="en">/);
  assert.match(html, /rel="canonical" href="https:\/\/trae1oung\.github\.io\/posttrainbench0\/"/);
  assert.match(html, /ScholarlyArticle/);
  assert.match(html, /Minzheng Wang/);
  assert.match(html, /Shizhu He/);
  assert.match(html, /Jun Zhao/);
  assert.match(html, /Kang Liu/);
  assert.match(html, /Institute of Automation, Chinese Academy of Sciences/);
  assert.match(html, /\/posttrainbench0\/assets\/index-/);
  assert.match(page, /useState<Language>\("en"\)/);
  assert.match(page, /中文/);
  assert.match(page, />EN</);
  assert.match(page, /Task definition and setup/);
  assert.match(page, /51 included complete agent runs/);
  assert.match(page, /PostTrainBench⁰ reformulates LLM post-training/);
  assert.doesNotMatch(page, /The conclusion of this blog is neither/);
  assert.doesNotMatch(page, /href="https:\/\/arxiv\.org\/abs\/2603\.08640">PostTrainBench/);
  assert.match(main, /createRoot/);
  assert.doesNotMatch(page, /GPT-5\.5 API/);
});

test("ships interactive data and checked research figures", async () => {
  const requiredAssets = [
    "figures/posttrainbench-system.png",
    "figures/premise-check.png",
    "figures/weight-neighborhood-and-task-directions.png",
    "figures/task-alignment-and-scale-sensitivity.png",
    "figures/historical-agent-runs.png",
    "figures/agent_model_4h_trajectory.png",
    "figures/agent-search-directions.png",
    "trajectory-data.json",
    "weight-space-data.json",
    "sitemap.xml",
  ];
  await Promise.all(requiredAssets.map((path) => access(new URL(`dist-pages/${path}`, root))));
  await Promise.all([
    access(new URL("docs/figures/posttrainbench0-system.drawio", root)),
    access(new URL("docs/figures/posttrainbench0-system.pdf", root)),
  ]);

  const bundles = await readdir(new URL("dist-pages/assets/", root));
  assert.ok(bundles.some((name) => /^index-.*\.js$/.test(name)));
  assert.ok(bundles.some((name) => /^index-.*\.css$/.test(name)));

  const [page, css, drawio, runtime, trace, prompt] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
    readFile(new URL("docs/figures/posttrainbench0-system.drawio", root), "utf8"),
    readFile(new URL("app/RuntimeContract.tsx", root), "utf8"),
    readFile(new URL("app/AgentTraceExplorer.tsx", root), "utf8"),
    readFile(new URL("public/prompt.txt", root), "utf8"),
  ]);
  assert.match(page, /Python source:/);
  assert.match(drawio, /Trusted Evaluator/);
  assert.match(drawio, /retain the best checkpoint/);
  assert.doesNotMatch(page, /src="[^"]+\.svg"/);
  assert.doesNotMatch(css, /Georgia|Times New Roman/);
  assert.match(css, /font-family:\s*-apple-system/);
  assert.match(runtime, /\/home\/agent/);
  assert.match(runtime, /Full instruction/);
  assert.match(trace, /Next step observed after feedback/);
  assert.match(trace, /Kimi K2\.6 · OpenCode/);
  assert.match(trace, /Claude Opus 4\.8 high · Cursor/);
  assert.match(trace, /Case conclusion/);
  assert.match(css, /--trace-columns/);
  assert.match(prompt, /The provided RandOpt and ES implementations/);
  assert.doesNotMatch(page, /Supported by current evidence/);
  assert.doesNotMatch(page, /Toward a more reliable evaluation/);
  assert.doesNotMatch(page, /Why we are releasing the experiment/);
  assert.doesNotMatch(page, /className="closing"/);
  assert.match(page, /Validity threats and limitations/);
  assert.match(page, /Every score reported here is a development score/);
  assert.match(page, /finite-budget zeroth-order search is strongly path-dependent/i);
  assert.match(page, /best viewed at this stage as a research prototype and perspective/);
  assert.match(page, /In one minute/);
});

test("uses one 0–100 score scale across interactive figures", async () => {
  const [trajectory, weight] = await Promise.all([
    readFile(new URL("app/InteractiveTrajectory.tsx", root), "utf8"),
    readFile(new URL("app/WeightSpaceExplorer.tsx", root), "utf8"),
  ]);
  assert.match(trajectory, /const displayScore = \(value: number\) => \(value \* 100\)\.toFixed\(2\)/);
  assert.match(trajectory, /Best seven-task mean so far \(0–100\)/);
  assert.match(trajectory, /Qwen2\.5-3B-Instruct · 34 complete runs/);
  assert.match(weight, /displayScore\(selected\.scores\.joint\)/);
});
