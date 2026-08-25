# Evaluation data

`visible200/` is the exact seven-task evaluation snapshot used by the public
PostTrainBench⁰ experiments. Each task contains 200 examples, selected as the
first 200 rows in the source order used when the snapshot was created. The
evaluator reads these files; they are never mounted into the Agent workspace.

| Task | Upstream source | File in this repository |
|---|---|---|
| Countdown | `Jiayi-Pan/Countdown-Tasks-3to4` | `countdown/countdown.json` |
| GSM8K | `openai/gsm8k`, train split | `gsm8k/train.parquet` |
| MATH-500 | `HuggingFaceH4/MATH-500`, test split | `math-500/test.jsonl` |
| OlympiadBench | `OpenBMB/OlympiadBench`, English math subset | `olympiadbench/OE_TO_maths_en_COMP.parquet` |
| MBPP | `google-research-datasets/mbpp`, sanitized split | `mbpp_full/` |
| ROCStories | ROCStories sentence-ordering data | `rocstories/train.parquet` |
| USPTO-50K | USPTO-50K reaction classification data | `uspto_50k/train.parquet` |

The files preserve the formats consumed by the pinned RandOPT data handlers.
`visible200/data_manifest.json` is the machine-readable layout. Run
`python scripts/check_environment.py --data-root data/visible200 --data-only`
to verify the snapshot without a GPU.

The upstream datasets retain their original licenses and citation
requirements. This repository republishes only the fixed evaluation subset
needed to execute the benchmark.
