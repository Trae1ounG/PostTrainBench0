import assert from "node:assert/strict";
import { access, readFile } from "node:fs/promises";
import test from "node:test";

async function render() {
  const workerUrl = new URL("../dist/server/index.js", import.meta.url);
  workerUrl.searchParams.set("test", `${process.pid}-${Date.now()}`);
  const { default: worker } = await import(workerUrl.href);
  return worker.fetch(
    new Request("http://localhost/", { headers: { accept: "text/html" } }),
    { ASSETS: { fetch: async () => new Response("Not found", { status: 404 }) } },
    { waitUntil() {}, passThroughOnException() {} },
  );
}

test("server-renders the PostTrainBench0 research article", async () => {
  const response = await render();
  assert.equal(response.status, 200);
  assert.match(response.headers.get("content-type") ?? "", /^text\/html\b/i);
  const html = await response.text();
  assert.match(html, /PostTrainBench<sup>0<\/sup>/);
  assert.match(html, /\{PostTrainBench\}\$\^\{0\}\$/);
  assert.match(html, /Task definition and setup/);
  assert.match(html, /Table of Contents/);
  assert.match(html, /1,400 inferences/);
  assert.match(html, /figures\/posttrainbench-system\.png/);
  assert.match(html, /figures\/historical-agent-runs\.png/);
  assert.match(html, /Run-by-run explorer/);
  assert.match(html, /Figure 7/);
  assert.match(html, /Table(?:\s|<!-- -->)*4/);
  assert.match(html, /class="katex"/);
  assert.match(html, /class="math-figure/);
  assert.match(html, /id="ref-1"/);
  assert.match(html, /7 × 200 = 1,400/);
  assert.match(html, /Rank, B\., Bhatnagar, H\./);
  assert.match(html, /Evolution Strategies at Scale/);
  assert.match(html, /RSIBench-Data/);
  assert.match(html, /id="ref-5"/);
  assert.match(html, /not strict RSI/);
  assert.match(html, /51 four-hour agent runs/);
  assert.match(html, /9\.90-point span/);
  assert.match(html, /Direction sampling/);
  assert.match(html, /Candidate count/);
  assert.match(html, /benchmark remains in validation/);
  assert.doesNotMatch(html, /Agentic ESOpt/);
  assert.doesNotMatch(html, /GPT-5\.5 API/);
  assert.doesNotMatch(html, /posttrain0bench-system-design-v3\.svg/);
});

test("ships the required figure assets and a plain sans-serif reading style", async () => {
  const requiredAssets = [
    "figures/posttrainbench-system.png",
    "figures/premise-check.png",
    "figures/weight-neighborhood-and-task-directions.png",
    "figures/task-alignment-and-scale-sensitivity.png",
    "figures/historical-agent-runs.png",
    "figures/agent_model_4h_trajectory.png",
    "figures/agent-search-directions.png",
  ];
  await Promise.all(requiredAssets.map((path) => access(new URL(`../public/${path}`, import.meta.url))));
  await Promise.all([
    access(new URL("../docs/figures/posttrainbench0-system.drawio", import.meta.url)),
    access(new URL("../docs/figures/posttrainbench0-system.pdf", import.meta.url)),
  ]);

  const [page, css, drawio] = await Promise.all([
    readFile(new URL("../app/page.tsx", import.meta.url), "utf8"),
    readFile(new URL("../app/globals.css", import.meta.url), "utf8"),
    readFile(new URL("../docs/figures/posttrainbench0-system.drawio", import.meta.url), "utf8"),
  ]);
  assert.match(page, /Python source:/);
  assert.match(drawio, /Trusted Evaluator/);
  assert.match(drawio, /retain the best checkpoint/);
  assert.doesNotMatch(page, /src="[^"]+\.svg"/);
  assert.doesNotMatch(css, /Georgia|Times New Roman/);
  assert.match(css, /font-family:\s*var\(--font-geist-sans\)/);
});
