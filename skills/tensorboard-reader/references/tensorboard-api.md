# How tbread reads TensorBoard logs (API notes)

`tbread` uses only the `tensorboard` pip package (no tensorflow). The relevant modules:

- `tensorboard.backend.event_processing.event_multiplexer.EventMultiplexer` — loads many
  runs at once.
- `tensorboard.backend.event_processing.io_wrapper.GetLogdirSubdirectories(logdir)` —
  finds every directory under `logdir` that contains event files. Used for discovery and
  to compute run names (path relative to `logdir`; the root maps to `(root)`).
- `tensorboard.util.tensor_util.make_ndarray(tensor_proto)` — decodes tensor-backed
  summaries (text, tensor-scalars, etc.) to numpy arrays.

## Discovery

Each directory containing event files becomes one run. Multiple event files in the same
dir (e.g. several training processes, or resumed runs) are merged by tag automatically.
`--run PAT` filters by run name/basename glob and only loads matching runs (faster on big
logdirs).

## Gotchas the reader handles for you

- **`SummaryMetadata(tag)` raises `KeyError` for legacy scalars.** Only tensor-backed tags
  have plugin metadata. The reader wraps the lookup and treats a missing/empty plugin name
  as "unknown".
- **Default `size_guidance` silently downsamples** (scalars→10k, tensors→10, images→4).
  That would corrupt counts and ranges. The reader builds the multiplexer with
  `size_guidance={type: 0}` (store everything) for accurate inventory, then does its own
  downsampling only at output time (`scalars --series --points N`).
- **Duplicate / non-monotonic steps.** Multi-process logs can write the same step from two
  processes. The reader dedupes by step (keeping the latest wall_time) and surfaces a
  `note: M duplicate steps merged`. Non-monotonic step sequences are flagged too.
- **Corrupt/partial trailing records** (common while training is live) are tolerated by the
  underlying directory watcher; a run that still fails to load is reported as
  `[unreadable: ...]` and the rest of the overview continues.

## Tag-type buckets

`accumulator.Tags()` returns: `scalars`, `tensors`, `histograms`, `distributions`,
`images`, `audio` (lists), plus `graph`/`meta_graph` (booleans). Accessors:
`Scalars(tag)`, `Tensors(tag)`, `Histograms(tag)`, `CompressedHistograms(tag)`,
`Images(tag)`, `Audio(tag)`, `PluginTagToContent(plugin)`, `SummaryMetadata(tag)`.
