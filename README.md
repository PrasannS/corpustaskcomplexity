# No More Free Lunch: Corpus Task Complexity Matters As Corpora Grow

Some corpus tasks like retrieval grow only linearly difficult in corpus size, but others grow quadratically difficulty (e.g. finding all contradictions in wikipedia). How do these behave differently when evaluating with long-context language models?

This repo includes code to reproduce experiments, as well as our training data and CTC-Bench evaluation suite!

**Browse real examples for CTC-Bench — every task, every rung:**
[corpus-reasoning-viz.pages.dev](https://corpus-reasoning-viz.pages.dev/) — gold documents
highlighted, the exact model prompt, the exact target string.

| resource | where |
|---|---|
| Eval ladders — 22 tasks × up to 10 rungs | [`PrasannSinghal/ctc-suite-eval`](https://huggingface.co/datasets/PrasannSinghal/ctc-suite-eval) (public, parquet) |
| Seed pools — the expensive half of generation, precomputed | [`PrasannSinghal/ctc-seed-pools`](https://huggingface.co/datasets/PrasannSinghal/ctc-seed-pools) (public) |
| Benchmark harness | [`allenai/olmo-eval`](https://github.com/allenai/olmo-eval), CTC tasks are all added in main branch|
---

## Install

```bash
git clone https://github.com/PrasannS/corpustaskcomplexity && cd corpustaskcomplexity

pip install ./ctc          # data generation + evaluation. No GPU, no CUDA, no compiler.
pip install -e '.[all]'    # the training side (OLMo-core). Install torch for your CUDA first.
```

**If you only want the data and the benchmark, the first
line is enough** — `ctc` imports no `olmo_core`, and the task JSONL it emits is a
framework-agnostic interface you can tokenize however you like.

```bash
pytest ctc/tests           # ~1250 tests, no GPU, no network, ~2 min
```

## Quickstart: from nothing to a graded number

```bash
# 1. build a training set and an eval ladder  (no GPU, no index, no API key)
ctc-data build --task contradiction --pool auto --train 18000 --out data/
ctc-data build --task contradiction --split eval --rungs 2k,8k,32k --out data/

# 2. tokenize to OLMo-core shards
PYTHONPATH=src:ctc/src python src/scripts/ctc/convert_to_shards.py \
    --input data/contradiction/train.jsonl --out shards/contradiction \
    --layout chunked --query-position after

# 3. fine-tune
CTC_NPROC=8 run/train.sh my-run --data shards/contradiction:1 \
    --base /path/to/qwen3.5-4b --arch chunked-mix --model qwen3_5_4B --tokenizer qwen3_5 \
    --lr 5e-5 --max-steps 7500

# 4. grade it
ctc-eval --ckpt runs/my-run/step7500 --tasks contradiction --bundle data/ --backend vllm
```

---

## 1. Data generation

Every corpus-backed task has a published **seed pool** — so that you don't need to recompute expensive steps
(cross-encoder scores, BM25 hard negatives, LLM-mined pairs).

```bash
ctc-data list                                                   # every task, its ladder, its knobs
ctc-data build --task nq --pool auto --train 20000 --out data/
ctc-data build --task textgroups --train 20000 --out data/      # synthetic: no pool needed
```

## 2. Tokenizing to shards

```bash
PYTHONPATH=src:ctc/src python src/scripts/ctc/convert_to_shards.py \
    --input data/contradiction/train.jsonl --out shards/contradiction \
    --layout chunked --query-position after
```

`--layout` and `--query-position` enter the fingerprint here. The flags that must be reproduced at
eval time — and which of them are checked automatically — are listed in
[`src/scripts/ctc/README.md`](src/scripts/ctc/README.md).

## 3. Training

One parameterised recipe, the experiment axes as flags — locally under `torchrun`, or on Beaker
with the same options plus `--cluster`.

```bash
CTC_NPROC=8 run/train.sh ctc-contradiction-chunked \
    --data shards/contradiction:1 --base BASE --arch chunked-mix \
    --model qwen3_5_4B --tokenizer qwen3_5 --lr 5e-5 --max-steps 7500
```

`--data DIR[:WEIGHT]` is repeatable and the weights are ratios, so `a:2 b:1` mixes 2:1. `--arch` is
one of:

- `full` — plain causal attention over the identical marker-bearing token stream.
- `chunked` — the document-chunked mask, strictly (p = 0 always).
- `chunked-mix` — the chunked mask **plus the mask-mixing curriculum** (each example collapses to
  plain causal with probability p, annealed 0.80 → 0.0 over the run). **This is the arm published
  "chunked" numbers use.** It is a different arm from `chunked`; don't relabel one as the other.
- `hierarchical`, `landmark` — the sparse-attention variants.

> ⚠ **A fresh Qwen3 base needs its marker embeddings repaired first.** Qwen3 never trains the
> reserved marker rows the document-chunked layout is built on, so they are bit-identical: the
> model cannot distinguish an open-document marker from a close one, and marker-dense training goes
> to chance in a way that reads exactly like a modeling result. Run
> `src/scripts/ctc/fix_marker_embeddings.py` (`--check-only` audits without writing). Qwen3.5 bases
> do **not** need this — audited at 0.8B/2B/4B/9B. The trainer's `--base` error text says so too.

## 4. Evaluation

### The olmo-eval CTC bench

We recommend using olmo-eval's CTC bench implementation for evaluation. The 22-row suite runs as a task family 
in AI2's eval harness against the public HF dataset — nothing from this repo required.

```bash
git clone -b prasann/ctc-suite-grader-fixes https://github.com/allenai/olmo-eval
cd olmo-eval

uv run olmo-eval run -m mock -t ctc_contradiction:r32k --dry-run   # preview, no model
uv run olmo-eval run -m <model> -t ctc_nq:r64k --save-predictions  # one task, one rung
```


## Reproducing the paper experiments

[`REPRODUCING.md`](REPRODUCING.md) gives the main experiments as standalone, node-local commands:
the 22-task dense-vs-chunked grid, the 0.8B/2B/4B model-scale sweep, and the 5-task mixed-SFT
family — with the reference hyperparameters recovered from the launch records.

## Layout

| where | what |
|---|---|
| [`ctc/`](ctc/) | **A self-contained pip package.** Task specs and the prompt/parse/metric contract (`ctc/format`), data generation (`ctc/data`), evaluation (`ctc/eval`). Imports no `olmo_core` except behind the `native` extra. Task JSONL is the boundary between the two halves. |
| [`src/scripts/ctc/`](src/scripts/ctc/) | The training side — everything that reads or writes OLMo-core formats: shard conversion, marker-embedding repair, the SFT/CPT recipe, Beaker launch. |
| [`run/`](run/) | `data.sh` → `convert.sh` → `train.sh` → `eval.sh`: thin wrappers that resolve the cluster environment first (interpreter, caches, `PYTHONPATH`). `_env.sh` encodes those traps once. |
| `src/olmo_core/` | Upstream OLMo-core plus our additions — the attention variants (document-chunked, landmark, hierarchical), composable instance sources, the format-fingerprint callback. |
| [`AGENTS.md`](AGENTS.md) | Orientation for coding agents, and the short list of things that bite. |

Golden fixtures under `ctc/tests/` were captured from the pre-migration implementation *before*
this tree was written, so they are independent evidence that the port preserves the numbers.
Regenerating one to make a test pass destroys that evidence; the fix is the code.

## Relation to OLMo-core

This repository is a fork of [allenai/OLMo-core](https://github.com/allenai/OLMo-core) — AI2's
training library — with the CTC suite added on top. Everything outside `ctc/`, `run/` and
`src/scripts/ctc/` is upstream OLMo-core; its own README is kept here as
[`README-OLMo-core.md`](README-OLMo-core.md) (installation detail, Docker images, code checks,
`make docs`), and the [API docs](https://olmo-core.readthedocs.io/en/latest/) still apply.
