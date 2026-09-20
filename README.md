# Corpus Task Complexity (CTC)

**22 long-context tasks whose difficulty scales with how much of an in-prompt corpus a model has to
track at once — with context ladders from 2k to 1M+ tokens, the generators that build them, and the
training code to fine-tune against them.**

A CTC task puts N documents in the prompt and asks a question that cannot be answered from any one
of them. The tasks span a complexity axis:

| class | what the answer requires | examples |
|---|---|---|
| **O(N)** | find the answer-bearing document — in principle solvable by a retriever | retrieval (NQ, FiQA, HotpotQA, SciFact, MS MARCO), NIAH, OOLONG, absence, outlier with fixed K |
| **O(N²)** | a *relation over* documents, with no single span to retrieve | every contradicting pair; query↔document matching; exact-copy orphans; restoring shuffled order; planted shared word-runs |
| **O(NM)** / **O(N³)** | structure over the whole corpus | cluster everything; find the planted feature-sum triple |

That axis scales *independently* of context length, which is the point: a model that merely
retrieves well separates from one that tracks a corpus long before either runs out of context
window.

**Browse real examples — every task, every rung:**
[corpus-reasoning-viz.pages.dev](https://corpus-reasoning-viz.pages.dev/) — gold documents
highlighted, the exact model prompt, the exact target string.

| resource | where |
|---|---|
| Eval ladders — 22 tasks × up to 10 rungs | [`PrasannSinghal/ctc-suite-eval`](https://huggingface.co/datasets/PrasannSinghal/ctc-suite-eval) (public, parquet) |
| Seed pools — the expensive half of generation, precomputed | [`PrasannSinghal/ctc-seed-pools`](https://huggingface.co/datasets/PrasannSinghal/ctc-seed-pools) (public) |
| Benchmark harness | [`allenai/olmo-eval`](https://github.com/allenai/olmo-eval), branch `prasann/ctc-suite-grader-fixes` |

---

## Install

```bash
git clone https://github.com/PrasannS/corpustaskcomplexity && cd corpustaskcomplexity

pip install ./ctc          # data generation + evaluation. No GPU, no CUDA, no compiler.
pip install -e '.[all]'    # the training side (OLMo-core). Install torch for your CUDA first.
```

The two halves are deliberately separable. **If you only want the data and the benchmark, the first
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

The four stages are threaded by a **format fingerprint** — recorded beside the shards at tokenize
time, written into every checkpoint at train time, checked at eval time. Eval refuses to grade a
checkpoint against a format it was not trained on. This is not decorative: reproducing one
pre-migration number cost two extra training runs to discover that `query_position` differed,
because nothing had recorded it.

---

## 1. Data generation

Every corpus-backed task has a published **seed pool** — the expensive half of generation
(cross-encoder scores, BM25 hard negatives, LLM-mined pairs) precomputed and shipped. `--pool auto`
fetches it from the Hub, so a 20k-example build takes about a minute and needs no GPU, no Lucene
index and no API key. The synthetic tasks need nothing at all.

```bash
ctc-data list                                                   # every task, its ladder, its knobs
ctc-data build --task nq --pool auto --train 20000 --out data/
ctc-data build --task textgroups --train 20000 --out data/      # synthetic: no pool needed

# rungs are open-ended past the calibrated 2k–32k table
ctc-data build --task textgroups --split eval --rungs 64k,1m,10m \
    --eval-size 125 --allow-small-eval --out data/xlong
```

Rungs above 32k are extrapolated from a least-squares fit through each task's own measured 2k–32k
table, and the ladder **refuses a rung its source corpus cannot supply** (qdmatch over HotpotQA
exhausts all 4,000 labeled units at 256k; strmatch is vocabulary-bound at 48k). Every build runs an
audit and writes nothing that fails it. Per-task supply bounds, corpus requirements and the
pool-export path are in [`ctc/src/ctc/data/README.md`](ctc/src/ctc/data/README.md).

**Coverage note.** 18 of the 22 suite rows have generators in this tree. The other four
(`msmarco`, `niah`, `obliq`, `qdmatch_fiqa`) were built with pre-migration pipelines and are served
ready-made in the public eval dataset.

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

The 22-row suite runs as a task family in AI2's eval harness against the public HF dataset —
nothing from this repo required.

```bash
git clone -b prasann/ctc-suite-grader-fixes https://github.com/allenai/olmo-eval
cd olmo-eval

uv run olmo-eval run -m mock -t ctc_contradiction:r32k --dry-run   # preview, no model
uv run olmo-eval run -m <model> -t ctc_nq:r64k --save-predictions  # one task, one rung
```

Suites cut the same 22 tasks along two orthogonal axes:

```bash
# by context length
uv run olmo-eval run -m <model> -t ctc:figure        # all 22 tasks, the 2k–32k grid  (108 runs)
uv run olmo-eval run -m <model> -t ctc:xlong         # everything above 32k            (69 runs)
uv run olmo-eval run -m <model> -t ctc:r128k         # every task at one rung

# by corpus-tracking demand — the axis the suite is named for
uv run olmo-eval run -m <model> -t ctc:low           # the 11 O(N) rows               (102 runs)
uv run olmo-eval run -m <model> -t ctc:high          # the 11 O(N²)+ rows              (75 runs)
uv run olmo-eval run -m <model> -t ctc:high:figure   # the two axes compose            (53 runs)
```

`ctc:high:figure` is the cheapest useful probe: grid-sized cost, and it is exactly where a model
that retrieves well separates from one that tracks a corpus. Prompt templates, parsers, metrics,
gold conventions and stop rules there are **vendored byte-faithful** from the `ctc` package in this
repo, so the harness and the generators cannot drift; fix things here and re-vendor.

Suite aggregation is DISPLAY_ONLY by design — the metrics are heterogeneous (f1, pair f1, Kendall
tau, `ce_pos_recall`, partial credit) and a cross-task mean would be meaningless.

### `ctc-eval` — grade your own checkpoint against your own bundles

```bash
ctc-eval --list-backends                    # what this install can run
ctc-eval --ckpt CKPT --tasks contradiction --bundle data/ --backend vllm
ctc-eval --ckpt CKPT --tasks contradiction --bundle data/ --attn chunked --backend native
```

Backends: `vllm` (fastest), `hf`, and `native` — the last grades an OLMo-core checkpoint directly
and is the only one that can apply the chunked/landmark masks. Install with the matching extra
(`pip install './ctc[vllm]'`). vLLM specifics, including the serving-copy requirement for
olmo-exported Qwen3.5 checkpoints, are in
[`ctc/src/ctc/eval/README.md`](ctc/src/ctc/eval/README.md).

### Reading the numbers

- **Score every arm with the same backend.** Native-vs-vLLM drift is ~0.08 f1 on contradiction@2k —
  larger than the eval's own standard error.
- **Quote `eval_size` and a standard error inline.** Rungs ≥256k hold 125 examples (SE ≈ ±0.041 at
  f1 ≈ 0.7); `scifact` is 300 and `obliq` 126 at every rung; everything else is 500. In this
  project `n` means *corpus size*, never eval-set size.
- **Check `parse_rate` before reading a score off a small model.** Sub-1B checkpoints often cannot
  answer in the suite's answer space at all, which floors every row below chance instead of ranking
  them: measured at r2k, the share of generations emitting any `[id]` is 3.5–13.5% at 450M, 5–61%
  at 810M, ~100% at 1.4B.
- **A rung label is a build target, not a measured length.** Labels come from the reference prompt
  path; files carry `_measured_prefill_tokens` where re-measured. Quote the measurement.

---

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
`make docs`), and the [API docs](https://olmo-core.readthedocs.io/en/latest/) still apply. Apache
2.0, as upstream.

## Citing

```bibtex
@misc{corpus-task-complexity,
  title  = {Corpus Task Complexity: long-context tasks that scale with corpus tracking},
  author = {Singhal, Prasann},
  year   = {2026},
  url    = {https://github.com/PrasannS/corpustaskcomplexity}
}
```

OLMo-core itself is [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656); that citation is in
[`README-OLMo-core.md`](README-OLMo-core.md).
