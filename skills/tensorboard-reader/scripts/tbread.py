#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["tensorboard>=2.12"]
# ///
"""tbread — read TensorBoard event logs with progressive detail, easy on tokens.

Tiers, cheapest first:
  overview   tiny map of runs -> tag inventory (counts/ranges, NEVER values)
  scalars    stats (default) or downsampled series for scalar tags
  text       newest-step text, truncated
  hist       histogram bucket summary at a step
  hparams    compact hparam table across runs
  info       shape/dtype/count/bytes for ANY tag (cheap bridge for binary)
  export     decode bytes to files on disk (the only command that writes bytes)

Binary types (images/audio/mesh/graph) are never inlined into stdout; use
`info` to size them and `export` to extract them.
"""
import argparse
import fnmatch
import json
import os
import sys
import time

RECOVERY = (
    "tensorboard not installed. Run with uv (recommended):\n"
    "  uv run scripts/tbread.py <args>\n"
    "or install it into the active environment:\n"
    "  pip install 'tensorboard>=2.12'"
)
try:
    from tensorboard.backend.event_processing import event_multiplexer
    from tensorboard.backend.event_processing import io_wrapper
    from tensorboard.util import tensor_util
except ImportError:
    print(RECOVERY, file=sys.stderr)
    sys.exit(3)

# event_accumulator tag-type bucket keys
T_SCALARS = "scalars"
T_TENSORS = "tensors"
T_HISTOGRAMS = "histograms"
T_DISTRIBUTIONS = "distributions"
T_IMAGES = "images"
T_AUDIO = "audio"
T_GRAPH = "graph"

# Normalized display types and whether they are binary/blob (never inlined).
BINARY_TYPES = {"images", "audio", "mesh", "graph"}

# Plugin name -> normalized type for tensor-backed tags.
PLUGIN_TYPE = {
    "scalars": "scalars",
    "text": "text",
    "pr_curves": "pr_curves",
    "mesh": "mesh",
    "hparams": "hparams",
    "images": "images",
    "audio": "audio",
    "histograms": "histograms",
}


# --------------------------------------------------------------------------- #
# Discovery
# --------------------------------------------------------------------------- #
def run_name(logdir, subdir):
    rel = os.path.relpath(subdir, logdir)
    return "(root)" if rel == "." else rel.replace(os.sep, "/")


def match_run(name, patterns):
    if not patterns:
        return True
    base = name.rsplit("/", 1)[-1]
    return any(fnmatch.fnmatch(name, p) or fnmatch.fnmatch(base, p) for p in patterns)


def load_multiplexer(logdir, run_patterns):
    """Discover run dirs, add only matching ones, reload. Returns (mux, [names])."""
    if not os.path.isdir(logdir):
        sys.exit(f"not a directory: {logdir}")
    everything = {k: 0 for k in (T_SCALARS, T_TENSORS, T_HISTOGRAMS,
                                 T_DISTRIBUTIONS, T_IMAGES, T_AUDIO)}
    mux = event_multiplexer.EventMultiplexer(size_guidance=everything)
    subdirs = sorted(io_wrapper.GetLogdirSubdirectories(logdir))
    names = []
    for sub in subdirs:
        name = run_name(logdir, sub)
        if not match_run(name, run_patterns):
            continue
        mux.AddRun(sub, name)
        names.append(name)
    if names:
        mux.Reload()
    return mux, sorted(names)


# --------------------------------------------------------------------------- #
# Classification & accessors
# --------------------------------------------------------------------------- #
def plugin_name(acc, tag):
    try:
        return acc.SummaryMetadata(tag).plugin_data.plugin_name or ""
    except (KeyError, Exception):  # legacy scalars raise KeyError
        return ""


def classify(acc):
    """Return {normalized_type: [tag, ...]} and a tag->meta map.

    meta[tag] = {"type": str, "tensor": bool, "display": str}
    """
    tags = acc.Tags()
    groups = {}
    meta = {}

    def add(ntype, tag, tensor, display=None):
        groups.setdefault(ntype, []).append(tag)
        meta[tag] = {"type": ntype, "tensor": tensor,
                     "display": display if display is not None else tag}

    for tag in tags.get(T_SCALARS, []):
        add("scalars", tag, False)
    for tag in tags.get(T_HISTOGRAMS, []):
        add("histograms", tag, False)
    for tag in tags.get(T_DISTRIBUTIONS, []):
        add("distributions", tag, False)
    for tag in tags.get(T_IMAGES, []):
        add("images", tag, False)
    for tag in tags.get(T_AUDIO, []):
        add("audio", tag, False)
    for tag in tags.get(T_TENSORS, []):
        pn = plugin_name(acc, tag)
        ntype = PLUGIN_TYPE.get(pn, "tensors")
        display = tag
        if ntype == "text" and display.endswith("/text_summary"):
            display = display[: -len("/text_summary")]
        add(ntype, tag, True, display)
    if tags.get(T_GRAPH):
        groups.setdefault("graph", [])  # presence marker
    return groups, meta


def scalar_series(acc, tag, is_tensor):
    """Return list of (step, wall_time, value) for a scalar/scalar-tensor tag."""
    out = []
    if is_tensor:
        for ev in acc.Tensors(tag):
            try:
                val = float(tensor_util.make_ndarray(ev.tensor_proto).reshape(-1)[0])
            except Exception:
                continue
            out.append((ev.step, ev.wall_time, val))
    else:
        for ev in acc.Scalars(tag):
            out.append((ev.step, ev.wall_time, ev.value))
    return out


def dedupe_steps(series):
    """Keep latest wall_time per step. Returns (clean, n_dupes, non_monotonic)."""
    by_step = {}
    dupes = 0
    for step, wall, val in series:
        if step in by_step:
            dupes += 1
            if wall >= by_step[step][1]:
                by_step[step] = (step, wall, val)
        else:
            by_step[step] = (step, wall, val)
    clean = sorted(by_step.values(), key=lambda r: r[0])
    steps = [r[0] for r in series]
    non_mono = any(b < a for a, b in zip(steps, steps[1:]))
    return clean, dupes, non_mono


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def hhmm(wall):
    return time.strftime("%H:%M", time.localtime(wall))


def fmtnum(x):
    if x is None:
        return "-"
    ax = abs(x)
    if x == int(x) and ax < 1e15:
        return str(int(x))
    if ax != 0 and (ax < 1e-3 or ax >= 1e6):
        return f"{x:.4g}"
    return f"{x:.4g}"


def stats_of(series):
    vals = [v for _, _, v in series]
    steps = [s for s, _, _ in series]
    n = len(vals)
    if n == 0:
        return {"count": 0}
    first, last = vals[0], vals[-1]
    mean = sum(vals) / n
    # simple least-squares slope over index for trend label
    if n >= 2:
        xs = list(range(n))
        mx = (n - 1) / 2
        my = mean
        denom = sum((x - mx) ** 2 for x in xs)
        slope = sum((x - mx) * (v - my) for x, v in zip(xs, vals)) / denom if denom else 0.0
        delta = last - first
        if abs(delta) < 1e-12:
            trend = "flat"
        else:
            arrow = "up" if delta > 0 else "down"
            trend = f"{arrow} {'+' if delta >= 0 else ''}{fmtnum(delta)}"
    else:
        slope, trend = 0.0, "flat"
    return {
        "count": n, "min": min(vals), "max": max(vals),
        "first": first, "last": last, "mean": mean,
        "step_min": steps[0], "step_max": steps[-1], "trend": trend,
        "slope": slope,
    }


def downsample(series, points, uniform):
    """Min/max-preserving bucket downsample, or uniform stride."""
    n = len(series)
    if n <= points:
        return series
    if uniform:
        stride = n / points
        idx = sorted({int(i * stride) for i in range(points)} | {n - 1})
        return [series[i] for i in idx]
    # min/max-preserving: always keep first & last; bucket the middle.
    nb = max(1, (points - 2) // 2)
    chosen = {0, n - 1}
    body = series[1:-1]
    if body:
        bsize = len(body) / nb
        for b in range(nb):
            lo = int(b * bsize)
            hi = int((b + 1) * bsize) if b < nb - 1 else len(body)
            if lo >= hi:
                continue
            seg = body[lo:hi]
            mn = min(range(len(seg)), key=lambda i: seg[i][2])
            mx = max(range(len(seg)), key=lambda i: seg[i][2])
            chosen.add(1 + lo + mn)
            chosen.add(1 + lo + mx)
    return [series[i] for i in sorted(chosen)]


def out_json(obj):
    print(json.dumps(obj, indent=2, default=str))


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_overview(args):
    mux, names = load_multiplexer(args.logdir, args.run)
    if not names:
        msg = f"no TensorBoard runs found under {args.logdir}"
        if args.json:
            out_json({"logdir": args.logdir, "runs": [], "note": msg})
        else:
            print(msg)
            print("(looked for directories containing events.out.tfevents.* files)")
        return 0

    types_filter = set(args.types.split(",")) if args.types else None
    jruns = []
    if not args.json and not args.quiet:
        print(f"logdir: {args.logdir}   runs={len(names)}")

    for name in names:
        acc = mux.GetAccumulator(name)
        try:
            groups, meta = classify(acc)
        except Exception as e:
            if args.json:
                jruns.append({"run": name, "unreadable": str(e)})
            else:
                print(f"RUN {name}   [unreadable: {e}]")
            continue

        jtypes = {}
        if not args.json:
            print(f"RUN {name}")
        for ntype in sorted(groups):
            if types_filter and ntype not in types_filter:
                continue
            tags = groups[ntype]
            if ntype == "graph":
                if not args.json:
                    print("  graph: present")
                jtypes["graph"] = {"present": True}
                continue
            entries = []
            for tag in tags:
                d = meta[tag]
                e = {"tag": d["display"]}
                if ntype in ("scalars",):
                    series = scalar_series(acc, tag, d["tensor"])
                    series, _, _ = dedupe_steps(series)
                    e["n"] = len(series)
                    if series:
                        e["step"] = [series[0][0], series[-1][0]]
                        e["wall"] = [series[0][1], series[-1][1]]
                else:
                    e["n"] = _count_events(acc, tag, d)
                entries.append((tag, e))

            jtypes[ntype] = [e for _, e in entries]
            if args.json:
                continue
            shown = entries[: args.max_tags]
            print(f"  {ntype}({len(entries)}):")
            for _, e in shown:
                if "step" in e:
                    s0, s1 = e["step"]
                    w0, w1 = e["wall"]
                    print(f"    {e['tag']:<32} n={e['n']:<5} "
                          f"step {s0}..{s1}   wall {hhmm(w0)}..{hhmm(w1)}")
                else:
                    print(f"    {e['tag']:<32} n={e['n']}")
            if len(entries) > args.max_tags:
                print(f"    (+{len(entries) - args.max_tags} more; "
                      f"use --max-tags or a drill-down command)")
        jruns.append({"run": name, "types": jtypes})

    if args.json:
        out_json({"logdir": args.logdir, "runs": jruns})
    return 0


def _count_events(acc, tag, meta):
    ntype, is_tensor = meta["type"], meta["tensor"]
    try:
        if ntype == "histograms":
            return len(acc.Histograms(tag))
        if ntype == "distributions":
            return len(acc.CompressedHistograms(tag))
        if ntype == "images":
            return len(acc.Images(tag))
        if ntype == "audio":
            return len(acc.Audio(tag))
        if is_tensor:
            return len(acc.Tensors(tag))
        return len(acc.Scalars(tag))
    except Exception:
        return -1


def _resolve_tags(acc, meta_map, groups, wanted, want_all, ntype):
    """Resolve --tag patterns (globs over display or raw) within a type."""
    pool = [(t, meta_map[t]["display"]) for t in groups.get(ntype, [])]
    if want_all:
        return [t for t, _ in pool]
    out = []
    for pat in wanted or []:
        for raw, disp in pool:
            if (fnmatch.fnmatch(disp, pat) or fnmatch.fnmatch(raw, pat)
                    or disp == pat or raw == pat):
                if raw not in out:
                    out.append(raw)
    return out


def cmd_scalars(args):
    mux, names = load_multiplexer(args.logdir, args.run)
    jout = []
    for name in names:
        acc = mux.GetAccumulator(name)
        groups, meta = classify(acc)
        tags = _resolve_tags(acc, meta, groups, args.tag, args.all, "scalars")
        for tag in tags:
            series = scalar_series(acc, tag, meta[tag]["tensor"])
            series, dupes, non_mono = dedupe_steps(series)
            disp = meta[tag]["display"]
            notes = []
            if dupes:
                notes.append(f"{dupes} duplicate steps merged")
            if non_mono:
                notes.append("non-monotonic steps")

            if args.series or args.full:
                pts = series if args.full else downsample(series, args.points, args.uniform)
                rec = {"run": name, "tag": disp, "n": len(series),
                       "emitted": len(pts),
                       "mode": "full" if args.full else
                               ("uniform" if args.uniform else "min/max-preserving"),
                       "points": [[s, v] for s, _, v in pts]}
                if notes:
                    rec["notes"] = notes
                jout.append(rec)
                if not args.json:
                    mode = rec["mode"]
                    print(f"{name}  {disp} (n={len(series)} -> {len(pts)} pts, {mode})")
                    print("  " + "  ".join(f"{s}:{fmtnum(v)}" for s, _, v in pts))
                    for nt in notes:
                        print(f"  note: {nt}")
            else:
                st = stats_of(series)
                rec = {"run": name, "tag": disp, **st}
                if notes:
                    rec["notes"] = notes
                jout.append(rec)
                if not args.json:
                    if st["count"] == 0:
                        print(f"{name}  {disp}  (no points)")
                        continue
                    print(f"{name}  {disp}  n={st['count']} "
                          f"min={fmtnum(st['min'])} max={fmtnum(st['max'])} "
                          f"first={fmtnum(st['first'])} last={fmtnum(st['last'])} "
                          f"mean={fmtnum(st['mean'])} trend={st['trend']}")
                    for nt in notes:
                        print(f"  note: {nt}")
    if not jout and not args.json:
        print("no matching scalar tags (try `overview` first)")
    if args.json:
        out_json(jout)
    return 0


def cmd_text(args):
    mux, names = load_multiplexer(args.logdir, args.run)
    jout = []
    for name in names:
        acc = mux.GetAccumulator(name)
        groups, meta = classify(acc)
        tags = _resolve_tags(acc, meta, groups, args.tag, args.all, "text")
        for tag in tags:
            events = acc.Tensors(tag)
            if not events:
                continue
            disp = meta[tag]["display"]
            picks = events if args.all_steps else [_pick_step(events, args.step)]
            for ev in picks:
                text = _decode_text(ev)
                truncated = len(text) > args.max_chars
                shown = text[: args.max_chars]
                rec = {"run": name, "tag": disp, "step": ev.step,
                       "chars": len(text), "truncated": truncated, "text": shown}
                jout.append(rec)
                if not args.json:
                    print(f"=== {name}  {disp}  step={ev.step}  ({len(text)} chars) ===")
                    print(shown)
                    if truncated:
                        print(f"... (+{len(text) - args.max_chars} chars; "
                              f"--max-chars N or `export` for full)")
    if not jout and not args.json:
        print("no matching text tags (try `overview` first)")
    if args.json:
        out_json(jout)
    return 0


def _pick_step(events, step):
    if step is None:
        return events[-1]
    return min(events, key=lambda e: abs(e.step - step))


def _decode_text(ev):
    arr = tensor_util.make_ndarray(ev.tensor_proto)
    flat = arr.reshape(-1)
    parts = []
    for x in flat:
        parts.append(x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x))
    return "\n".join(parts)


def cmd_hist(args):
    mux, names = load_multiplexer(args.logdir, args.run)
    jout = []
    for name in names:
        acc = mux.GetAccumulator(name)
        groups, meta = classify(acc)
        tags = _resolve_tags(acc, meta, groups, args.tag, args.all, "histograms")
        for tag in tags:
            events = acc.Histograms(tag)
            if not events:
                continue
            ev = _pick_step(events, args.step)
            h = ev.histogram_value
            summary = _hist_summary(h, args.bins)
            rec = {"run": name, "tag": tag, "step": ev.step,
                   "min": h.min, "max": h.max, "num": h.num,
                   "sum": h.sum, "mean": (h.sum / h.num if h.num else 0.0),
                   "std": _hist_std(h), "buckets": summary}
            jout.append(rec)
            if not args.json:
                print(f"{name}  {tag}  step={ev.step}  "
                      f"num={fmtnum(h.num)} min={fmtnum(h.min)} max={fmtnum(h.max)} "
                      f"mean={fmtnum(rec['mean'])} std={fmtnum(rec['std'])}")
                for lo, hi, cnt in summary:
                    print(f"  [{fmtnum(lo)}, {fmtnum(hi)}) : {fmtnum(cnt)}")
    if not jout and not args.json:
        print("no matching histogram tags (try `overview` first)")
    if args.json:
        out_json(jout)
    return 0


def _hist_std(h):
    if not h.num:
        return 0.0
    mean = h.sum / h.num
    var = max(0.0, h.sum_squares / h.num - mean * mean)
    return var ** 0.5


def _hist_summary(h, nbins):
    """Re-bin the fine TB buckets into ~nbins coarse [lo,hi):count bins."""
    limits, counts = list(h.bucket_limit), list(h.bucket)
    if not limits:
        return []
    lo, hi = h.min, h.max
    if hi <= lo:
        return [(lo, hi, sum(counts))]
    width = (hi - lo) / nbins
    coarse = [0.0] * nbins
    prev = lo
    for lim, cnt in zip(limits, counts):
        center = (prev + lim) / 2
        prev = lim
        idx = int((center - lo) / width)
        idx = min(max(idx, 0), nbins - 1)
        coarse[idx] += cnt
    return [(lo + i * width, lo + (i + 1) * width, coarse[i]) for i in range(nbins)]


def cmd_hparams(args):
    mux, names = load_multiplexer(args.logdir, args.run)
    rows = {}
    for name in names:
        acc = mux.GetAccumulator(name)
        try:
            content = acc.PluginTagToContent("hparams")
        except (KeyError, Exception):
            content = {}
        if not content:
            continue
        from tensorboard.plugins.hparams import plugin_data_pb2
        for tag, raw in content.items():
            try:
                data = plugin_data_pb2.HParamsPluginData.FromString(raw)
            except Exception:
                continue
            sess = data.session_start_info
            for k, v in sess.hparams.items():
                rows.setdefault(name, {})[k] = (
                    v.number_value or v.string_value or v.bool_value)
    if args.json:
        out_json(rows)
    elif not rows:
        print("no hparams found")
    else:
        keys = sorted({k for r in rows.values() for k in r})
        for name in names:
            if name in rows:
                print(f"RUN {name}")
                for k in keys:
                    if k in rows[name]:
                        print(f"  {k} = {rows[name][k]}")
    return 0


def cmd_info(args):
    mux, names = load_multiplexer(args.logdir, args.run)
    jout = []
    for name in names:
        acc = mux.GetAccumulator(name)
        groups, meta = classify(acc)
        # search every type for the tag pattern
        all_tags = list(meta.keys())
        matched = []
        for pat in args.tag:
            for t in all_tags:
                if (fnmatch.fnmatch(meta[t]["display"], pat)
                        or fnmatch.fnmatch(t, pat) or t == pat):
                    if t not in matched:
                        matched.append(t)
        for tag in matched:
            d = meta[tag]
            info = _tag_info(acc, tag, d)
            info.update({"run": name, "tag": d["display"], "type": d["type"]})
            jout.append(info)
            if not args.json:
                extra = " ".join(f"{k}={v}" for k, v in info.items()
                                 if k not in ("run", "tag", "type"))
                print(f"{name}  {d['display']}  type={d['type']}  {extra}")
    if not jout and not args.json:
        print("no matching tags (try `overview` first)")
    if args.json:
        out_json(jout)
    return 0


def _tag_info(acc, tag, meta):
    ntype = meta["type"]
    try:
        if ntype == "images":
            evs = acc.Images(tag)
            tot = sum(len(e.encoded_image_string) for e in evs)
            wh = (evs[-1].width, evs[-1].height) if evs else (0, 0)
            return {"count": len(evs), "width": wh[0], "height": wh[1],
                    "total_bytes": tot}
        if ntype == "audio":
            evs = acc.Audio(tag)
            tot = sum(len(e.encoded_audio_string) for e in evs)
            sr = evs[-1].sample_rate if evs else 0
            return {"count": len(evs), "sample_rate": sr, "total_bytes": tot}
        if ntype == "histograms":
            return {"count": len(acc.Histograms(tag))}
        if ntype == "distributions":
            return {"count": len(acc.CompressedHistograms(tag))}
        if meta["tensor"]:
            evs = acc.Tensors(tag)
            shape = dtype = None
            tot = 0
            for e in evs:
                tot += len(e.tensor_proto.SerializeToString())
            if evs:
                arr = tensor_util.make_ndarray(evs[-1].tensor_proto)
                shape, dtype = list(arr.shape), str(arr.dtype)
            return {"count": len(evs), "shape": shape, "dtype": dtype,
                    "total_bytes": tot, "plugin": plugin_name(acc, tag)}
        evs = acc.Scalars(tag)
        return {"count": len(evs)}
    except Exception as e:
        return {"error": str(e)}


def cmd_export(args):
    mux, names = load_multiplexer(args.logdir, args.run)
    os.makedirs(args.out, exist_ok=True)
    written = []
    for name in names:
        acc = mux.GetAccumulator(name)
        groups, meta = classify(acc)
        all_tags = list(meta.keys())
        matched = [t for t in all_tags
                   if any(fnmatch.fnmatch(meta[t]["display"], p)
                          or fnmatch.fnmatch(t, p) or t == p for p in args.tag)]
        for tag in matched:
            written += _export_tag(acc, tag, meta[tag], name, args)
    total = sum(w[1] for w in written)
    if args.json:
        out_json({"out": args.out,
                  "files": [{"path": p, "bytes": b} for p, b in written]})
    else:
        for p, b in written:
            print(f"  {b:>10} B  {p}")
        print(f"wrote {len(written)} file(s), {total} B -> {args.out}")
    return 0


def _safe(name):
    return name.replace("/", "_").replace(" ", "_")


def _write(path, data, written):
    mode = "wb" if isinstance(data, (bytes, bytearray)) else "w"
    with open(path, mode) as f:
        f.write(data)
    written.append((path, os.path.getsize(path)))


def _export_tag(acc, tag, meta, run, args):
    ntype = meta["type"]
    base = os.path.join(args.out, f"{_safe(run)}__{_safe(meta['display'])}")
    written = []
    try:
        if ntype == "images":
            evs = acc.Images(tag)
            sel = evs if args.all else evs[-1:]
            for e in sel:
                _write(f"{base}__step{e.step}.png", e.encoded_image_string, written)
        elif ntype == "audio":
            evs = acc.Audio(tag)
            sel = evs if args.all else evs[-1:]
            for e in sel:
                _write(f"{base}__step{e.step}.wav", e.encoded_audio_string, written)
        elif ntype == "text":
            evs = acc.Tensors(tag)
            sel = evs if args.all else evs[-1:]
            for e in sel:
                _write(f"{base}__step{e.step}.txt", _decode_text(e), written)
        elif ntype == "scalars":
            series = scalar_series(acc, tag, meta["tensor"])
            series, _, _ = dedupe_steps(series)
            lines = ["step,wall_time,value"] + [f"{s},{w},{v}" for s, w, v in series]
            _write(f"{base}.csv", "\n".join(lines) + "\n", written)
        elif ntype == "histograms":
            evs = acc.Histograms(tag)
            sel = evs if args.all else evs[-1:]
            for e in sel:
                h = e.histogram_value
                lines = ["bucket_limit,count"] + [
                    f"{lim},{cnt}" for lim, cnt in zip(h.bucket_limit, h.bucket)]
                _write(f"{base}__step{e.step}.csv", "\n".join(lines) + "\n", written)
        elif meta["tensor"]:
            evs = acc.Tensors(tag)
            sel = evs if args.all else evs[-1:]
            for e in sel:
                arr = tensor_util.make_ndarray(e.tensor_proto)
                _write(f"{base}__step{e.step}.txt", repr(arr.tolist()), written)
        else:
            written.append((f"<unsupported export type: {ntype}>", 0))
    except Exception as e:
        written.append((f"<error exporting {tag}: {e}>", 0))
    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser():
    p = argparse.ArgumentParser(
        prog="tbread",
        description="Read TensorBoard logs with progressive detail (token-friendly).")
    sub = p.add_subparsers(dest="cmd")

    def common(sp):
        sp.add_argument("logdir", help="directory containing TensorBoard runs")
        sp.add_argument("--run", action="append", default=[],
                        help="restrict to run(s); glob, repeatable")
        sp.add_argument("--json", action="store_true", help="machine-readable JSON")
        sp.add_argument("--quiet", action="store_true", help="suppress banner")

    ov = sub.add_parser("overview", help="tiny map of runs and tag inventory")
    common(ov)
    ov.add_argument("--max-tags", type=int, default=40)
    ov.add_argument("--types", help="comma-separated type filter")
    ov.set_defaults(func=cmd_overview)

    sc = sub.add_parser("scalars", help="scalar stats or downsampled series")
    common(sc)
    sc.add_argument("--tag", action="append", default=[], help="tag glob, repeatable")
    sc.add_argument("--all", action="store_true", help="all scalar tags")
    sc.add_argument("--series", action="store_true", help="emit downsampled series")
    sc.add_argument("--points", type=int, default=50, help="series point cap")
    sc.add_argument("--uniform", action="store_true", help="uniform stride sampling")
    sc.add_argument("--full", action="store_true", help="emit ALL raw points")
    sc.set_defaults(func=cmd_scalars)

    tx = sub.add_parser("text", help="newest-step text, truncated")
    common(tx)
    tx.add_argument("--tag", action="append", default=[])
    tx.add_argument("--all", action="store_true")
    tx.add_argument("--step", type=int, default=None)
    tx.add_argument("--all-steps", action="store_true")
    tx.add_argument("--max-chars", type=int, default=2000)
    tx.set_defaults(func=cmd_text)

    hi = sub.add_parser("hist", help="histogram bucket summary at a step")
    common(hi)
    hi.add_argument("--tag", action="append", default=[])
    hi.add_argument("--all", action="store_true")
    hi.add_argument("--step", type=int, default=None, help="default: latest")
    hi.add_argument("--bins", type=int, default=8)
    hi.set_defaults(func=cmd_hist)

    hp = sub.add_parser("hparams", help="compact hparam table across runs")
    common(hp)
    hp.set_defaults(func=cmd_hparams)

    inf = sub.add_parser("info", help="shape/dtype/count/bytes for ANY tag")
    common(inf)
    inf.add_argument("--tag", action="append", default=[], required=True)
    inf.set_defaults(func=cmd_info)

    ex = sub.add_parser("export", help="decode bytes to files on disk")
    common(ex)
    ex.add_argument("--tag", action="append", default=[], required=True)
    ex.add_argument("--out", required=True, help="output directory")
    ex.add_argument("--all", action="store_true", help="all steps (default: latest)")
    ex.set_defaults(func=cmd_export)

    return p


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    # default subcommand: overview
    parser = build_parser()
    known = {"overview", "scalars", "text", "hist", "hparams", "info", "export"}
    if argv and argv[0] not in known and not argv[0].startswith("-"):
        argv = ["overview"] + argv
    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    try:
        return args.func(args)
    except BrokenPipeError:
        return 0


if __name__ == "__main__":
    sys.exit(main())
