#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["tensorboard>=2.12", "numpy"]
# ///
"""End-to-end harness: generate logs with tensorboard, parse them with tbread.py.

Run it:
    uv run tests/test_tbread.py            # generate in a temp dir, assert, clean up
    uv run tests/test_tbread.py --keep DIR # also leave the generated logs in DIR

Generation and parsing share one `tensorboard` install, so `sys.executable`
(the interpreter running this harness) is used to invoke tbread.py — no nested uv.
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TBREAD = os.path.join(HERE, "..", "skills", "tensorboard-reader", "scripts", "tbread.py")

sys.path.insert(0, HERE)
import gen_logs  # noqa: E402

_failures = []
_passed = 0


def check(name, cond, detail=""):
    global _passed
    if cond:
        _passed += 1
        print(f"  PASS  {name}")
    else:
        _failures.append((name, detail))
        print(f"  FAIL  {name}  {detail}")


def run(*args, expect_code=0):
    """Invoke tbread.py; return (stdout, stderr, code)."""
    proc = subprocess.run([sys.executable, TBREAD, *args],
                          capture_output=True, text=True)
    if proc.returncode != expect_code:
        check(f"exit code for {' '.join(args[:2])}", False,
              f"got {proc.returncode}, want {expect_code}; stderr={proc.stderr[-300:]}")
    return proc.stdout, proc.stderr, proc.returncode


def jrun(*args):
    out, err, code = run(*args, "--json")
    try:
        return json.loads(out)
    except json.JSONDecodeError as e:
        check(f"valid json for {' '.join(args[:2])}", False, f"{e}; out={out[:300]}")
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keep", help="keep generated logs in this directory")
    args = ap.parse_args()

    tmp = args.keep or tempfile.mkdtemp(prefix="tb-synth-")
    os.makedirs(tmp, exist_ok=True)
    gen_logs.generate(tmp)
    print(f"generated synthetic logs at {tmp}\n")

    # ---- overview: discovery + inventory --------------------------------------
    ov = jrun("overview", tmp)
    runs = {r["run"]: r for r in (ov or {}).get("runs", [])}
    check("overview discovers 3 non-empty runs", set(runs) == {"runs/seg1", "tb_logs", "multi"},
          str(set(runs)))
    seg1 = runs.get("runs/seg1", {}).get("types", {})
    check("seg1 scalars present", "scalars" in seg1)
    seg1_scalar_tags = {e["tag"] for e in seg1.get("scalars", [])}
    check("tensor-scalar reclassified as scalar (train/score)",
          "train/score" in seg1_scalar_tags, str(seg1_scalar_tags))
    loss_entry = next((e for e in seg1.get("scalars", []) if e["tag"] == "train/loss"), {})
    check("train/loss n=40", loss_entry.get("n") == 40, str(loss_entry))
    check("train/loss step range 0..390", loss_entry.get("step") == [0, 390], str(loss_entry))
    seg1_text = {e["tag"] for e in seg1.get("text", [])}
    check("text display strips /text_summary (config)", "config" in seg1_text, str(seg1_text))

    tb = runs.get("tb_logs", {}).get("types", {})
    check("tb_logs has image type", "images" in tb, str(list(tb)))
    check("tb_logs has audio type", "audio" in tb, str(list(tb)))
    check("tb_logs has histograms type", "histograms" in tb, str(list(tb)))

    # binary types must NOT carry step/wall value fields in overview, only counts
    img_entry = tb.get("images", [{}])[0]
    check("image overview is count-only (no values inlined)",
          set(img_entry) <= {"tag", "n"} and img_entry.get("n") == 1, str(img_entry))

    # ---- scalars stats + trend + spike ---------------------------------------
    st = jrun("scalars", tmp, "--run", "seg1", "--tag", "train/loss")
    rec = (st or [{}])[0]
    check("loss stats count=40", rec.get("count") == 40, str(rec))
    check("loss trend is down", str(rec.get("trend", "")).startswith("down"), str(rec))
    check("loss max captures spike (==5.0)", abs(rec.get("max", 0) - 5.0) < 1e-6, str(rec))

    ser = jrun("scalars", tmp, "--run", "seg1", "--tag", "train/loss",
               "--series", "--points", "10")
    pts = (ser or [{}])[0].get("points", [])
    vals = [v for _, v in pts]
    check("series emits <= ~12 points", 0 < len(pts) <= 12, str(len(pts)))
    check("series preserves spike value 5.0", any(abs(v - 5.0) < 1e-6 for v in vals),
          str(vals))
    steps = [s for s, _ in pts]
    check("series keeps first & last step", steps and steps[0] == 0 and steps[-1] == 390,
          str(steps))

    # ---- text drill-down ------------------------------------------------------
    tx = jrun("text", tmp, "--run", "tb_logs", "--tag", "eval/sample")
    trec = (tx or [{}])[0]
    check("text newest step == 300", trec.get("step") == 300, str(trec))
    check("text content matches", trec.get("text") == "sample at step 300", str(trec))

    txt = jrun("text", tmp, "--run", "tb_logs", "--tag", "eval/sample",
               "--max-chars", "5")
    check("text truncation flagged", (txt or [{}])[0].get("truncated") is True, str(txt))

    # ---- histogram summary ----------------------------------------------------
    h = jrun("hist", tmp, "--run", "tb_logs", "--tag", "weights/layer0", "--bins", "8")
    hrec = (h or [{}])[0]
    check("hist num=1000", hrec.get("num") == 1000, str(hrec))
    check("hist has 8 coarse bins", len(hrec.get("buckets", [])) == 8, str(hrec))
    check("hist bin counts sum ~1000",
          abs(sum(b[2] for b in hrec.get("buckets", [])) - 1000) < 1e-3, str(hrec))

    # ---- info on binary tags --------------------------------------------------
    info = jrun("info", tmp, "--run", "tb_logs", "--tag", "media/img")
    irec = next((r for r in (info or []) if r.get("type") == "images"), {})
    check("info image width/height/count", irec.get("count") == 1
          and irec.get("width") == 1 and irec.get("height") == 1, str(irec))
    check("info image reports bytes", irec.get("total_bytes", 0) > 0, str(irec))
    ainfo = jrun("info", tmp, "--run", "tb_logs", "--tag", "media/clip")
    arec = next((r for r in (ainfo or []) if r.get("type") == "audio"), {})
    check("info audio sample_rate=16000", arec.get("sample_rate") == 16000, str(arec))

    # ---- export writes files to disk, prints only manifest --------------------
    exp_dir = os.path.join(tmp, "_export")
    edata = jrun("export", tmp, "--run", "tb_logs",
                 "--tag", "media/img", "--tag", "eval/acc", "--tag", "eval/sample",
                 "--out", exp_dir)
    files = {os.path.basename(f["path"]) for f in (edata or {}).get("files", [])}
    check("export wrote a .png", any(f.endswith(".png") for f in files), str(files))
    check("export wrote a .csv", any(f.endswith(".csv") for f in files), str(files))
    check("export wrote a .txt", any(f.endswith(".txt") for f in files), str(files))
    check("exported files exist on disk and are non-empty",
          all(os.path.getsize(f["path"]) > 0 for f in (edata or {}).get("files", [])),
          str(files))

    # ---- dedup note on overlapping multi-process steps ------------------------
    md = jrun("scalars", tmp, "--run", "multi", "--tag", "proc/metric")
    mrec = (md or [{}])[0]
    # writer A wrote steps {0,10,20}, B wrote {10,20,30} -> 4 unique, 2 merged
    check("multi dedups to 4 unique steps", mrec.get("count") == 4, str(mrec))
    notes = " ".join(mrec.get("notes", []))
    check("multi surfaces duplicate-steps note", "duplicate" in notes, str(notes))

    # ---- empty / missing handling --------------------------------------------
    empty_only = jrun("overview", os.path.join(tmp, "empty"))
    check("empty dir -> no runs", empty_only.get("runs") == [], str(empty_only))
    _, _, code = run("overview", "/tmp/tbread-does-not-exist-xyz", expect_code=1)
    check("nonexistent dir exits 1", code == 1)

    # ---- no binary bytes ever leak into a value command -----------------------
    raw_overview, _, _ = run("overview", tmp)
    check("PNG header bytes never appear in overview text",
          "\x89PNG" not in raw_overview and "PNG\r\n" not in raw_overview)

    # ---- summary --------------------------------------------------------------
    print(f"\n{_passed} passed, {len(_failures)} failed")
    if _failures:
        for n, d in _failures:
            print(f"  - {n}: {d}")
        if not args.keep:
            _cleanup(tmp)
        return 1
    if not args.keep:
        _cleanup(tmp)
    print("ALL GREEN")
    return 0


def _cleanup(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
