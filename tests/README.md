# Tests

End-to-end harness: generate TensorBoard logs across many data types using only the
`tensorboard` package (no TensorFlow/torch), then parse them with the skill's `tbread.py`
and assert on the results.

```bash
uv run tests/test_tbread.py             # generate in a temp dir, assert, clean up
uv run tests/test_tbread.py --keep DIR  # also leave the generated logs in DIR to inspect
```

- `gen_logs.py` — writes a synthetic logdir: training scalars (with a loss spike), a
  tensor-backed scalar, text (with and without `/text_summary` suffix), an image, audio,
  a histogram, a two-file run with overlapping steps (dedup), and an empty dir.
- `test_tbread.py` — runs `overview / scalars / text / hist / info / export` against the
  generated logs and checks counts, ranges, trend, spike-preserving downsampling, text
  truncation, histogram re-binning, binary `info`, on-disk export, multi-process dedup
  notes, and empty/nonexistent-dir handling.

These live outside `skills/` so they are not part of the published skill payload.
