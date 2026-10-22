# RandMSD

A compact point-cloud semantic-segmentation backbone built on random-sampling
hierarchies. Local difference mixing combines neighborhood-scale geometry and
projected feature differences; KNN-max support transfer and an additive,
post-mixer-skip decoder propagate these features across resolutions.

**Source release candidate.** Licensed under the [MIT License](LICENSE), with
the upstream copyright notice retained in [NOTICE.txt](NOTICE.txt) and
[LICENSES/UPSTREAM-MIT.txt](LICENSES/UPSTREAM-MIT.txt). Pretrained weights and
datasets are not bundled. A formal paper citation will be added when available.

## Contents

| Component | Location |
|---|---|
| Public model and strict checkpoint loading | `src/randmsd/` |
| Frozen reference and memory-optimized backbone | `src/rsnative_corelite/` |
| Prepared-cloud conversion and serial coverage sampling | `src/randmsd/data/`, `preprocess.py` |
| Training, resume, sampled validation, full-cloud voting | `train.py`, `evaluate.py`, `engine.py` |
| Dataset recipes | `configs/` |
| CPU correctness and recovery tests | `tests/` |
| Model-only timing and pooled six-fold metrics | `benchmarks/`, `tools/` |

| Configuration | Attributes | Points/patch | Encoder widths | Parameters |
|---|---|---:|---|---:|
| S3DIS | XYZ + RGB | 40,960 | 32, 64, 128, 256, 384 | 1,976,381 |
| SemanticKITTI | XYZ + remission | 45,056 | 32, 64, 128, 256 | 754,307 |
| SensatUrban | XYZ + RGB | 65,536 | 32, 64, 128, 256, 384 | 1,976,381 |

The model consumes preconstructed `xyz`, `neigh_idx`, `sub_idx`, and
`interp_idx` lists plus `features[B,N,C]`. It returns `logits[B,classes,N]`.
The public aliases preserve historical state-dict keys. Both implementations
use the same parameters; optimized inference chunks center points, while
training optionally checkpoints the first local mixer.

## Install and test

Use a dedicated Python 3.10+ environment. Install a PyTorch build appropriate
for your CPU/CUDA platform, then install this project:

```bash
python -m pip install -e .
python -B -m unittest discover -s tests -v
python -B examples/model_smoke.py --dataset S3DIS
python -B examples/model_smoke.py --dataset SemanticKITTI
python -B examples/model_smoke.py --dataset SensatUrban
```

Runtime dependencies are PyTorch, NumPy, and SciPy. No custom extension or
third-party segmentation repository is required. The local release checks use
a CPU runtime; they do not replace CUDA or full-dataset validation. Optional
historical-code parity tests require `RANDMSD_LEGACY_ROOT` pointing to an
independently retained research checkout; otherwise those tests are skipped.

## Prepare existing subsampled datasets

The converter imports **existing preprocessed grids**, not raw dataset
downloads. It produces safe NPZ clouds and a checksummed manifest. Obtain and
prepare datasets under their own access terms. No data are downloaded by these
commands.

```bash
python -m randmsd.preprocess --dataset S3DIS \
  --source /data/S3DIS/input_0.040 --output data/s3dis_area5 \
  --test-area 5 --include-projections --trust-pickle

python -m randmsd.preprocess --dataset SemanticKITTI \
  --source /data/SemanticKITTI/dataset/sequences_0.06 \
  --raw-kitti /data/SemanticKITTI/dataset/sequences \
  --output data/kitti --include-projections --trust-pickle

python -m randmsd.preprocess --dataset SensatUrban \
  --source /data/SensatUrban/grid_0.200_author_recovered \
  --output data/sensaturban --include-projections --trust-pickle
```

Only use `--trust-pickle` with independently trusted local projection files:
pickle can execute arbitrary code. KDTree pickles are not loaded. Omitting
projections allows grid-level evaluation, but not original-resolution scores
or benchmark submission export. SemanticKITTI must include aligned
`remissions/*.npy`; labels are mapped to 19 contiguous training IDs, with
unlabeled points ignored. RGB is normalized once to [0,1].

S3DIS holds out the specified area. SemanticKITTI uses 00–07/09/10 for
training, 08 for validation, and 11–21 for test. SensatUrban recognizes the
fixed four validation tiles and test tiles; the complete paper training split
requires the author-recovered 33 training tiles. The converter does not
recover missing/corrupted tiles. Inspect the resulting manifest before use.

## Train and resume

```bash
python -m randmsd.train --config configs/s3dis.json \
  --data-root data/s3dis_area5 --output runs/s3dis_area5 --device cuda:0

python -m randmsd.train --config configs/s3dis.json \
  --data-root data/s3dis_area5 --output runs/s3dis_area5 --device cuda:0 --resume
```

Use `configs/semantickitti.json` or `configs/sensaturban.json` with the
corresponding prepared directory for the other datasets. The serial coverage
sampler uses `num_workers=0`. Start on an idle GPU and inspect memory before
raising batch sizes. Use `--device cpu` for small functional checks.

The output contains `config.json`, `history.json`, `best.pt`, `last.pt`, and
`training_result.json`. Best-checkpoint selection uses sampled-validation
mIoU. Resume checks the configuration, source hashes, data manifest, optimizer,
sampler, and RNG states; incompatible or incomplete historical state is not
silently ignored. Nonempty runs require an explicit resume request.

## Full-cloud voting and export

```bash
python -m randmsd.evaluate --config configs/s3dis.json \
  --data-root data/s3dis_area5 --checkpoint runs/s3dis_area5/best.pt \
  --output results/s3dis_area5 --device cuda:0 --save-predictions
```

`metrics.json` identifies whether the score is on the prepared grid or on
original points. `confusion.npz` stores both matrices. A coverage vote is a
possibility threshold, **not exactly one visit per point**. An insufficient
patch budget fails instead of reporting a partial score. Validation results
are separate from official hidden-test results.

For test predictions add `--split test`. Saved NPZ labels are contiguous
training IDs; export them to benchmark IDs and formats explicitly:

```bash
python -m randmsd.data.export --dataset SemanticKITTI \
  --predictions results/kitti_test \
  --raw-root /data/SemanticKITTI/dataset/sequences --output exports/kitti

python -m randmsd.data.export --dataset SensatUrban \
  --predictions results/sensaturban_test \
  --raw-root /data/SensatUrban/original_ply --output exports/sensaturban
```

SemanticKITTI export verifies the complete official test set and writes
`sequences/XX/predictions/*.label` as uint32 original semantic IDs with zero
instance bits. SensatUrban export verifies all six test tiles and writes
uint8 `.label` files plus `submission.zip`. Export checks original point
counts, rejects invalid labels, and never submits anything to a server.
Checkpoints load with restricted deserialization by default; use
`--trusted-checkpoint` only for a trusted legacy archive that needs it.

For S3DIS six-fold evaluation, prepare and train six independent held-out-area
directories. Pool confusion matrices, not per-area mIoUs:

```bash
python tools/aggregate_folds.py --space original --output results/sixfold.json \
  --confusions results/area1/confusion.npz results/area2/confusion.npz \
  results/area3/confusion.npz results/area4/confusion.npz \
  results/area5/confusion.npz results/area6/confusion.npz
```

## Resource measurements

```bash
python benchmarks/benchmark_forward.py --dataset SemanticKITTI \
  --device cuda:0 --batch-size 1 --output results/kitti_forward.json
```

This measures only the model forward pass on a **preconstructed synthetic
hierarchy**, after warmup. It excludes sampling, KNN, data transfer, scene
fusion, original-point projection, and I/O. Its result is not total scene
processing time. The manuscript's multi-model scene-cost study has additional
dependencies and a separate fixed-workload protocol; it is not bundled as an
unverified generic speed leaderboard.

## Reproduction status

The frozen model, parameter counts, strict weight loading, local weighted-loss
and schedule formulas, small CPU training/voting, and interrupted-run recovery
are tested. The clean data workflow uses SciPy KNN and its own serial RNG;
distance-tie ordering and patch traversal can differ from the historical
compiled pipeline. In particular, SemanticKITTI fixed-step training samples
scans with replacement rather than replaying the historical shuffled loader.

The JSON recipes retain the principal paper settings but are **not a claim
that the reorganized training pipeline already reproduces the paper scores**.
SensatUrban's five-coverage-vote run is also distinct from the historical
1,000-batch official-test budget. Raw-data voxelization, clean-pipeline CUDA
validation, full-dataset/checkpoint voting comparison, pretrained weight
distribution, and the formal paper citation remain pending.

