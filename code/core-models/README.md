# CORE BERT baselines

This repository trains encoder classifiers on the closed-label subset of the CORE intervention records.

## First task

The initial task predicts the gold answer after an intervention using four labels:

- `no`
- `yes`
- `less`
- `more`

The compatible sources are CLadder, WIQA, and the CCR.GB feasibility sample. Com2 is excluded from this BERT classification baseline because its answers are open text rather than a stable class vocabulary.

Current compatible split sizes are 20,963 train, 4,706 validation, and 1,918 held-out test records. The test split is not used during model or configuration selection.

## Commands

```bash
python3 -m unittest discover -s tests -v
CUDA_VISIBLE_DEVICES=0,1,2,3 bash scripts/run_bert_base.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash scripts/run_bert_large.sh
GPU_COUNT=16 CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 bash scripts/run_bert_large.sh
python3 scripts/validate_scope.py
```

Independent graph seeds are run concurrently because graph is the statistical unit. The matrix launcher uses exactly four GPUs for BERT-base and eight for BERT-large, assigning one graph/seed job to each GPU and replenishing the queue as jobs finish:

```bash
python3 scripts/launch_operator_matrix.py --phase foundation --model-size base \
  --gpus 0,1,2,3 --run-ids 101,102,103,104 --output-root outputs/base-foundations
```

Use `--phase operator-smoke` for the fast canonical O2/O3 forward/backward integration check. Operator runs use the compatible server runtime at `${PRIVATE_STORAGE_ROOT}/DLclass/bin/python`; the classifier-only DDP runner retains its newer isolated environment.

Run paths and model caches default to local ignored directories. Every training invocation writes its complete arguments, environment, data counts, baseline metrics, epoch metrics, and final metrics to `outputs/<run-name>/run_summary.json`.

T2-b and T3-a/T3-b addressing primitives are implemented in `src/core_bert/t2_inline.py` and `src/core_bert/t3_pointer.py`; `addressed_data.py` and `addressed_reader.py` provide benchmark collation and split-BERT integration. Their validation boundary and remaining DDP/CLI gate are recorded in `reports/T2_T3_IMPLEMENTATION.md`.

Run the addressed BERT smoke on exactly four GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 scripts/launch_addressed_smoke.py \
  --mode t2b --data-root data \
  --output-json outputs/addressed-t2b-smoke/run_summary.json
```

Use `--editor-disabled` for A2. For T3, use `--span-override correct`, `adjacent`, or `random` for A3. The smoke loader cannot open the held-out test split.

The production addressed-training path is `scripts/launch_addressed_training.py`. It uses four GPUs for BERT-base and eight for BERT-large, authoritative graph/world sidecars, variable-aligned slots, post-edit question conditioning, exact non-padding distributed validation, and validation-selected resumable checkpoints. Two-edit, unseen-paraphrase, and per-variable protocol modules are present, but authoritative truth/catalog artifacts and composed execution are not yet connected; runs therefore remain intentionally marked `a1_evidence: false`.

## Scale sequence

1. `google-bert/bert-base-uncased` (~110M parameters)
2. `google-bert/bert-large-uncased` (~340M parameters)
3. A larger encoder only after BERT-large improves validation metrics and memory/runtime are measured

The staged runs use DDP: four GPUs for BERT-base and eight GPUs for BERT-large. Sixteen-GPU BERT-large is selected only when measured fixed-workload throughput exceeds eight GPUs.

## Current result

BERT-base training is complete. The validation-selected epoch-3 checkpoint achieved 56.23% validation accuracy and 56.26% held-out test accuracy. A one-epoch BERT-large scale check achieved only 50.11% validation accuracy, so it was not evaluated on the held-out test set. See `reports/BERT_BASELINE.md` for full metrics and provenance.
