# TensorBoard data types

How `tbread` handles each TensorBoard data type across the three tiers. The reader
classifies tags from the event accumulator's type buckets, then refines tensor-backed tags
by their plugin name (`text`, `scalars`, `pr_curves`, `mesh`, `hparams`, ...).

| Type | Overview (Tier 0) | Drill-down (Tier 1) | Export (Tier 2) |
|------|-------------------|---------------------|-----------------|
| **scalars** | tag, n, step range, wall range | `scalars --stats`: count/min/max/first/last/mean/trend. `scalars --series`: downsampled curve (min/max-preserving, cap `--points`). | `.csv` of `step,wall_time,value` |
| **text** | tag, n | `text`: newest step, decoded, truncated to `--max-chars`. `--step`/`--all-steps` to pick. | `.txt` per step |
| **histograms** | tag, n | `hist`: num/min/max/mean/std + ~`--bins` coarse `[lo,hi):count` bins. Never raw bucket arrays. | `.csv` of `bucket_limit,count` per step |
| **distributions** | tag, n | counted; surfaced via `info`. | (compressed-histogram CSV) |
| **hparams** | listed | `hparams`: name→value table across runs. | — |
| **images** | tag, n only | `info`: width, height, count, total bytes. **No pixels inlined.** | `.png` per step |
| **audio** | tag, n only | `info`: sample_rate, count, total bytes. **No audio inlined.** | `.wav` per step |
| **mesh** | tag, n only | `info`: shape/dtype/bytes. | tensor dump |
| **pr_curves** | tag, n | `info` (shape); full curve via export. | tensor dump |
| **graph** | `graph: present` | `info` (when addressable). | — |
| **tensors** (other/unknown plugin) | tag, n | `info`: shape/dtype/bytes/plugin. | text dump of the ndarray |

## Notes on classification

- **Scalars logged as tensors.** Modern TensorFlow logs scalars as rank-0 tensors with
  plugin `scalars`. The reader reclassifies these into the scalar handlers, so you get a
  unified scalar view regardless of how they were written.
- **Text tags** usually carry a `/text_summary` suffix in the raw tag; the reader strips
  it for display (`args/text_summary` → `args`). `--tag` matches either form.
- **Binary guard.** images/audio/mesh/graph are flagged binary and physically routed away
  from any stdout value path. Use `info` to size them and `export` to extract them.

## Sizing a heavy tag before extracting

```bash
uv run scripts/tbread.py info LOGDIR --tag 'samples/*'   # how big? how many?
uv run scripts/tbread.py export LOGDIR --tag 'samples/*' --out /tmp/imgs --all
```
