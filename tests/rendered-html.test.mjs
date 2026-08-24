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
  assert.match(html, /\/posttrainbench0\/assets\/index-/);
  assert.match(page, /useState<Language>\("en"\)/);
  assert.match(page, /中文/);
  assert.match(page, />EN</);
  assert.match(page, /Task definition and setup/);
  assert.match(page, /51 four-hour agent runs/);
  assert.match(page, /benchmark remains in validation/);
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

  const [page, css, drawio] = await Promise.all([
    readFile(new URL("app/page.tsx", root), "utf8"),
    readFile(new URL("app/globals.css", root), "utf8"),
    readFile(new URL("docs/figures/posttrainbench0-system.drawio", root), "utf8"),
  ]);
  assert.match(page, /Python source:/);
  assert.match(drawio, /Trusted Evaluator/);
  assert.match(drawio, /retain the best checkpoint/);
  assert.doesNotMatch(page, /src="[^"]+\.svg"/);
  assert.doesNotMatch(css, /Georgia|Times New Roman/);
  assert.match(css, /font-family:\s*-apple-system/);
});
