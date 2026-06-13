# tensorboard-reader-skill

A skill and Python code to help agents read TensorBoard logs without setting tokens on fire.

It reads TensorBoard event logs (`.tfevents`) with **progressive detail**: a tiny overview
first, drilling down only where needed. Binary blobs (images/audio) and full scalar series
are never dumped into context unless you explicitly ask. Uses only the `tensorboard` pip
package (no TensorFlow).

## Quick start

```bash
# overview: runs -> tag inventory (counts/ranges, no values)
uv run skills/tensorboard-reader/scripts/tbread.py overview <LOGDIR>

# scalar stats, then a downsampled curve only if you need it
uv run skills/tensorboard-reader/scripts/tbread.py scalars <LOGDIR> --tag 'train/loss'
uv run skills/tensorboard-reader/scripts/tbread.py scalars <LOGDIR> --tag 'train/loss' --series --points 30
```

`uv` reads the script's inline dependency and runs it in an isolated env — no install
needed. Without `uv`, run with any Python that has `tensorboard` (`pip install -r
skills/tensorboard-reader/scripts/requirements.txt`).

## Commands

`overview` · `scalars` · `text` · `hist` · `hparams` · `info` · `export` — see
[skills/tensorboard-reader/SKILL.md](skills/tensorboard-reader/SKILL.md) for the full
workflow and [references/](skills/tensorboard-reader/references/) for data-type and API
details.

## Install as a skill

```bash
gh skill install jacobjennings/tensorboard-reader-skill
```

## License

Apache-2.0 — see [LICENSE](LICENSE). (Matches `tensorboard`, this skill's only
dependency; `tbread.py` imports it rather than vendoring any of its source.)
