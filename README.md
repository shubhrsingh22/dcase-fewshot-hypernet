# Hypernetworks for Few-Shot Bioacoustic Event Detection (DCASE Task 5)

Reproduction code for the few-shot bioacoustic detection experiments of thesis
Chapter 3.3 ("CNN Hypernetwork for Few-shot Bioacoustic Event Detection"): a
**Prototypical Network** baseline and **graph-hypernetwork augmented** variants
(**H-Proto**) that modulate the CNN feature maps with context-aware channel
gating.

The pipeline is built on the official DCASE few-shot bioacoustic baseline
(`c4dm/dcase-few-shot-bioacoustic`, the `deep_learning` ProtoNet system) with the
thesis modifications, and integrates the official event-based evaluation metric.

## Models

| `experiment_name` | `model.hyper_placement` | Description |
|---|---|---|
| `proto`      | `[]`        | Prototypical Network baseline |
| `hproto1`    | `[1]`       | Graph hypernetwork after conv block 1 |
| `hproto2`    | `[2]`       | after block 2 |
| `hproto3`    | `[3]`       | after block 3 |
| `hproto_all` | `[1,2,3,4]` | after every block |

- **Encoder** (`model.encoder`): `protonet` (thesis: 4 conv blocks of 64 filters,
  3×3, BN, ReLU, 2×2 max-pool → adaptive pool → linear to a 256-d embedding) or
  `resnet` (the DCASE baseline ResNet-9, 512-d embedding).
- **Graph-hypernetwork** (`src/model.py::GraphHyperNet`): treats the `H·W`
  feature-map locations as nodes, builds a k-NN graph (`model.hyper_knn`), runs
  two GCN layers (`C→C/2→C/4`), mean-pools, and maps to sigmoid channel-gating
  coefficients that modulate the feature map.

## Setup

```bash
conda create -y -n fsbio python=3.11 && conda activate fsbio
pip install -r requirements.txt
```

## Data

Download the DCASE Development Set (Zenodo record 10829604) and point
`config.path.data_root` at it (default `/path/to/Development_Set`):

```
Development_Set/
  Training_Set/{BV,HT,JD,MT,WMW}/*.{wav,csv}
  Validation_Set/{HB,ME,PB,PB24,PW,RD}/*.{wav,csv}
```

## Configuration

All settings live in `config/config.yaml` and are overridable from the CLI with
OmegaConf dotlist syntax, e.g. `python -m src.train model.encoder=resnet seed=7`.
Key knobs:

- `features.feature_type`: `pcen` (default; the baseline's feature) or `logmel`.
- `model.encoder`: `protonet` | `resnet`; `model.hyper_placement`: `[]`/`[3]`/...
- `train.contrast`: hard-negative contrastive training (episodes include
  in-recording background as distractor prototypes).
- `eval.neg_estimate`: `random` (sample negatives from unlabelled audio) or
  `energy` (RMS-based background detection).
- `eval.post_proc`: `fixed` (drop events < 200 ms) or `adaptive` (< a fraction of
  the shortest support shot).

## 1. Feature extraction (once, shared by all runs)

```bash
python -m src.train stage.features=true stage.train=false stage.eval=false
# -> features/train/Mel_train.h5 and features/eval/<recording>.h5
```

## 2. Train + evaluate a single model

```bash
# ProtoNet baseline
python -m src.train experiment_name=proto  model.hyper_placement="[]"  seed=42
# H-Proto-3
python -m src.train experiment_name=hproto3 model.hyper_placement="[3]" seed=42
```

Each run writes `logs/<experiment>/seed_<seed>/{best_model.pth,
Eval_out_postproc.csv, results.json}`.

## 3. Multi-seed study

```bash
# seeds, experiments, GPU ids
scripts/run_experiments.sh "42 1337 2024" "proto hproto1 hproto2 hproto3 hproto_all" "0"
python -m src.aggregate_results --logs_dir logs --out_dir results
```

## 4. Baselines / analysis utilities

```bash
# random-guessing floor
python -m src.random_baseline seed=42 path.work_dir=logs/random/seed_42
# detection-threshold sweep for a trained model
python -m src.sweep_threshold path.work_dir=logs/proto/seed_42 thresholds="0.5,0.6,0.7,0.8"
# best operating point (sweep threshold x post-processor)
python -m src.eval_best experiment_name=proto path.work_dir=logs/proto/seed_42
```

## Parameter count and walltime

Measured with `scripts/params_walltime.py` (single NVIDIA L40S, PyTorch 2.5.1,
float32, no JIT/torch.compile). Train step = one 5-way 5-shot episode with 5
queries/class (50 segments of 17 frames x 128 mels), prototypical loss,
forward+backward+Adam; inference on 128 segments under `no_grad`:

| Model | Trainable params (M) | Train episode (ms) | Inference (ms/segment) |
|---|---|---|---|
| proto      | 0.178 | 2.97 | 0.006 |
| hproto1    | 0.181 | 4.29 | 0.010 |
| hproto2    | 0.181 | 4.23 | 0.010 |
| hproto3    | 0.181 | 4.23 | 0.009 |
| hproto_all | 0.193 | 7.93 | 0.023 |

The graph hypernetwork adds ~2% parameters per placement but 40-170% step
time, dominated by the k-NN graph construction and dense adjacency ops.
Raw numbers and environment details: `results/params_walltime.{md,csv}`.

## Evaluation protocol

Event-based F-measure (`evaluation/`): a predicted event matches a reference POS
event when IoU ≥ 0.3, resolved by Hopcroft–Karp bipartite matching; predictions
matching UNK events are ignored. Precision/recall/F are computed per subset and
the **harmonic mean across subsets** is reported.

## Reproduction notes

- Features: the thesis text describes log-mel; the GitHub baseline uses PCEN
  (default here). Both are available via `features.feature_type`.
- The 4-conv ProtoNet uses an adaptive pool before the linear layer to fix the
  256-d embedding for the variable segment lengths used at validation time.
- Optimiser: Adam, lr 1e-3, StepLR ×0.65 every 10 epochs, early stopping on
  validation accuracy.
- Validation subsets differ across DCASE releases; scores are release-dependent,
  so evaluate on a fixed subset selection when comparing.

## Acknowledgements

Built on the DCASE few-shot bioacoustic baseline
(`c4dm/dcase-few-shot-bioacoustic`) and its official `evaluation_metrics`.
