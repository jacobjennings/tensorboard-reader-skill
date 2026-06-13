# Worked example: a HuggingFace Trainer run

A typical TRL/HF-Trainer checkpoint dir contains a `tb_logs/` directory (eval metrics +
status/text) and a `runs/` directory with one nested subdir per training segment. Here is
the progressive-detail flow against such a logdir.

## 1. Overview — the tiny map

```bash
uv run scripts/tbread.py overview checkpoints/run-XXXX/
```
```
logdir: checkpoints/run-XXXX/   runs=6
RUN runs/Jun12_02-17-05_host
  scalars(9):
    train/loss              n=72   step 10..720   wall 02:20..06:17
    train/grad_norm         n=72   step 10..720   wall 02:20..06:17
    train/learning_rate     n=72   step 10..720   wall 02:20..06:17
    ...
  text(2):
    args                    n=1
    model_config            n=1
RUN tb_logs
  scalars(4):
    eval/validity_rate          n=4   step 0..2864   wall 02:13..21:27
    eval/mean_chamfer_distance  n=4   step 0..2864   wall 02:13..21:27
    ...
  text(2):
    eval/sample_completion  n=4
    status/message          n=5
```

The training scalars live in the `runs/Jun12_*` segments; eval metrics and text samples
live in `tb_logs`. Now you know what exists without having read a single value.

## 2. Scalar stats — is loss going down?

```bash
uv run scripts/tbread.py scalars checkpoints/run-XXXX/ --tag 'train/loss'
```
```
runs/Jun12_02-17-05_host  train/loss  n=72 min=0.0035 max=1.373 first=1.373 last=0.0037 mean=0.069 trend=down -1.369
runs/Jun12_07-25-16_host  train/loss  n=71 min=0.0029 max=0.0060 first=0.0038 last=0.0036 mean=0.0038 trend=down -0.0002
...
```

Loss collapses in the first segment and plateaus low afterward — visible in one line per
run, no curve needed.

## 3. Series — only when you need the shape

```bash
uv run scripts/tbread.py scalars checkpoints/run-XXXX/ \
  --run 'Jun12_02-17-05*' --tag 'train/loss' --series --points 12
```
```
train/loss (n=72 -> 12 pts, min/max-preserving)
  10:1.373  30:0.81  ...  720:0.0037
```

Downsampling preserves the min and max in each bucket so a loss spike won't be hidden.

## 4. Text — inspect a sample completion

```bash
uv run scripts/tbread.py text checkpoints/run-XXXX/ \
  --run tb_logs --tag 'eval/sample_completion' --max-chars 400
```
Prints the newest-step text, truncated, with a `(+N chars; ... or export for full)` hint.

## 5. Export — pull the full artifact to disk

```bash
uv run scripts/tbread.py export checkpoints/run-XXXX/ \
  --run tb_logs --tag 'eval/validity_rate' --tag 'eval/sample_completion' \
  --out /tmp/tb-export
```
```
       191 B  /tmp/tb-export/tb_logs__eval_validity_rate.csv
       379 B  /tmp/tb-export/tb_logs__eval_sample_completion__step2864.txt
wrote 2 file(s), 570 B -> /tmp/tb-export
```

Bytes go to disk, never into context. Read the file directly if you need the full content.
