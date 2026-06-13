---
name: tensorboard-reader
description: Read and summarize TensorBoard event logs (.tfevents) token-efficiently. Use when inspecting a training/eval run's logs — scalars (loss, lr, metrics), text samples, histograms, hparams, or images/audio — without dumping raw data into context. Start with a tiny overview, then drill down only where needed.
license: Apache-2.0
---

# TensorBoard Reader

Inspect TensorBoard logs with **progressive detail**: a cheap overview first, drilling
down only where it matters. The reader never dumps full series or binary blobs unless you
explicitly ask, so reading a run costs a few hundred tokens instead of tens of thousands.

All commands go through one script: `scripts/tbread.py`.

## Quick start

```bash
uv run scripts/tbread.py overview <LOGDIR>
```

`<LOGDIR>` is a directory containing TensorBoard runs. Run discovery is recursive — it
finds nested run subdirs (e.g. `runs/Jun12_*`) and merges the many event files a single
dir may contain (multi-process training) into one logical run. `overview` is the default,
so `uv run scripts/tbread.py <LOGDIR>` works too.

## Progressive-detail workflow (always overview first)

| Tier | Command | What it costs / gives |
|------|---------|-----------------------|
| 0 | `overview LOGDIR` | Tiny map: runs → tags grouped by type, with point counts + step/wall ranges. **Never prints values.** |
| 1 | `scalars LOGDIR --tag T` | Default `--stats`: count/min/max/first/last/mean/trend. Add `--series` for a downsampled curve. |
| 1 | `text LOGDIR --tag T` | Newest-step text, truncated to `--max-chars` (default 2000). |
| 1 | `hist LOGDIR --tag T` | Histogram bucket summary at a step (coarse bins, not raw arrays). |
| 1 | `hparams LOGDIR` | Compact hparam table across runs. |
| 1 | `info LOGDIR --tag T` | Shape/dtype/count/bytes for **any** tag — the cheap way to size an image/audio/mesh tag before extracting it. |
| 2 | `export LOGDIR --tag T --out DIR` | Decodes bytes to files on disk (.png/.wav/.csv/.txt/...). The only command that writes bytes. |

Rule of thumb: run `overview`, decide which tags matter, then pull just those with
`scalars --stats`. Only reach for `--series`, `text`, or `export` when stats aren't enough.

## Common flags

- `--tag PAT` — glob over tag names; repeatable (`--tag 'train/*' --tag eval/loss`).
- `--run PAT` — restrict to run(s) by name or basename glob; repeatable. Cheaper too (only matching runs are loaded).
- `--json` — machine-readable output for every command.
- `scalars`: `--series` (downsampled curve), `--points N` (cap, default 50, min/max-preserving so spikes survive), `--uniform` (plain stride), `--full` (every raw point — opt-in), `--all` (all scalar tags).
- `text`: `--step N` (default newest), `--all-steps`, `--max-chars N`.
- `hist`: `--step N` (default latest), `--bins N` (default 8).
- `export`: `--out DIR` (required), `--all` (every step; default latest only).

## Output contract

- Binary/heavy types (**images, audio, mesh, graph**) are never inlined. `overview` and
  `info` report counts/shape/bytes; use `export` to get the actual files.
- `overview` collapses any type's tag list past `--max-tags` (default 40) to `(+N more)`.
- Notes like `duplicate steps merged` or `non-monotonic steps` are surfaced so you don't
  misread merged multi-process logs.

## Runtime & fallback

1. **Preferred:** `uv run scripts/tbread.py …` — `uv` reads the script's inline
   dependency (`tensorboard`) and runs it in an isolated, cached env. No global install,
   no tensorflow needed.
2. If `uv` is absent: `python3 scripts/tbread.py …` in an env that has tensorboard. On a
   missing import the script exits non-zero with the exact recovery command (it never
   auto-installs). `scripts/requirements.txt` pins the one dependency.

## References

- `references/data-types.md` — what each TensorBoard data type shows at overview vs drill-down vs export.
- `references/tensorboard-api.md` — how the reader works and the TensorBoard API gotchas it handles.
- `references/examples.md` — a worked HuggingFace-Trainer walkthrough.
