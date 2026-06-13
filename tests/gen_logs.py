#!/usr/bin/env python3
# /// script
# requires-python = ">=3.9"
# dependencies = ["tensorboard>=2.12", "numpy"]
# ///
"""Generate a synthetic TensorBoard logdir covering many data types.

Uses ONLY the `tensorboard` package (no TensorFlow / torch) so the generator
shares the skill's single dependency. Writes event files directly via
EventFileWriter + summary protos.

Layout produced under <out>:
  runs/seg1/         legacy scalars (loss w/ spike, lr), tensor-scalar metric, text
  tb_logs/           eval scalars, text(+/text_summary), image, audio, histogram
  multi/             two event files, overlapping steps (dedup test)
  empty/             no event files (no-runs test)
"""
import base64
import os
import struct
import sys
import time

import numpy as np
from tensorboard.compat.proto import event_pb2, summary_pb2
from tensorboard.plugins.scalar import metadata as scalar_metadata
from tensorboard.plugins.text import metadata as text_metadata
from tensorboard.summary.writer.event_file_writer import EventFileWriter
from tensorboard.util import tensor_util

# A real 1x1 PNG and a tiny mono 16-bit WAV, so image/audio bytes are valid.
PNG_1x1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAC0lEQVR4nGNg"
    "YGAAAAAEAAH2FzhVAAAAAElFTkSuQmCC")


def wav_bytes(samples=8, rate=16000):
    data = b"".join(struct.pack("<h", int(2000 * np.sin(i))) for i in range(samples))
    n = len(data)
    hdr = b"RIFF" + struct.pack("<I", 36 + n) + b"WAVE"
    hdr += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    hdr += b"data" + struct.pack("<I", n)
    return hdr + data


WAV = wav_bytes()
T0 = 1_700_000_000.0  # fixed wall-time base for reproducible ranges


def _writer(path, suffix=""):
    os.makedirs(path, exist_ok=True)
    return EventFileWriter(path, filename_suffix=suffix)


def _emit(w, step, value, wall):
    e = event_pb2.Event(step=step, wall_time=wall,
                        summary=summary_pb2.Summary(value=[value]))
    w.add_event(e)


def v_scalar(tag, x):
    return summary_pb2.Summary.Value(tag=tag, simple_value=float(x))


def v_tensor_scalar(tag, x):
    tp = tensor_util.make_tensor_proto(np.array(x, dtype=np.float32))
    meta = scalar_metadata.create_summary_metadata("", "")
    return summary_pb2.Summary.Value(tag=tag, tensor=tp, metadata=meta)


def v_text(tag, s):
    tp = tensor_util.make_tensor_proto(np.array([s.encode("utf-8")], dtype=object))
    meta = text_metadata.create_summary_metadata("", "")
    return summary_pb2.Summary.Value(tag=tag, tensor=tp, metadata=meta)


def v_histogram(tag, data, bins=30):
    counts, edges = np.histogram(data, bins=bins)
    hp = summary_pb2.HistogramProto(
        min=float(data.min()), max=float(data.max()), num=int(data.size),
        sum=float(data.sum()), sum_squares=float((data ** 2).sum()))
    hp.bucket_limit.extend(edges[1:].tolist())
    hp.bucket.extend(counts.tolist())
    return summary_pb2.Summary.Value(tag=tag, histo=hp)


def v_image(tag, png, w, h):
    img = summary_pb2.Summary.Image(height=h, width=w, colorspace=3,
                                    encoded_image_string=png)
    return summary_pb2.Summary.Value(tag=tag, image=img)


def v_audio(tag, wav, rate=16000):
    au = summary_pb2.Summary.Audio(sample_rate=rate, num_channels=1,
                                   length_frames=8, encoded_audio_string=wav,
                                   content_type="audio/wav")
    return summary_pb2.Summary.Value(tag=tag, audio=au)


def generate(out):
    # ---- runs/seg1: training scalars (loss with a mid spike), lr, tensor metric, text
    w = _writer(os.path.join(out, "runs", "seg1"))
    n = 40
    for i in range(n):
        step = i * 10
        wall = T0 + step
        loss = 2.0 * (1 - i / n) + 0.05
        if i == 20:
            loss = 5.0  # spike that min/max-preserving downsampling must keep
        _emit(w, step, v_scalar("train/loss", loss), wall)
        _emit(w, step, v_scalar("train/lr", 1e-3 * (1 - i / n)), wall)
        _emit(w, step, v_tensor_scalar("train/score", i / n), wall)
    _emit(w, 0, v_text("config/text_summary", "lr=1e-3\nbatch=8\nepochs=3"), T0)
    w.flush(); w.close()

    # ---- tb_logs: eval scalars, text, image, audio, histogram
    w = _writer(os.path.join(out, "tb_logs"))
    for i, step in enumerate((0, 100, 200, 300)):
        _emit(w, step, v_scalar("eval/acc", 0.5 + 0.1 * i), T0 + step)
        _emit(w, step, v_text("eval/sample/text_summary", f"sample at step {step}"),
              T0 + step)
    _emit(w, 300, v_image("media/img", PNG_1x1, 1, 1), T0 + 300)
    _emit(w, 300, v_audio("media/clip", WAV), T0 + 300)
    rng = np.random.default_rng(0)
    _emit(w, 300, v_histogram("weights/layer0", rng.normal(size=1000)), T0 + 300)
    w.flush(); w.close()

    # ---- multi: two event files in ONE dir with overlapping steps (dedup test)
    wa = _writer(os.path.join(out, "multi"), suffix=".a")
    for step in (0, 10, 20):
        _emit(wa, step, v_scalar("proc/metric", step / 10.0), T0 + step)
    wa.flush(); wa.close()
    wb = _writer(os.path.join(out, "multi"), suffix=".b")
    for step in (10, 20, 30):  # 10,20 overlap with writer A
        _emit(wb, step, v_scalar("proc/metric", step / 10.0 + 0.01), T0 + 100 + step)
    wb.flush(); wb.close()

    # ---- empty: a dir with no event files
    os.makedirs(os.path.join(out, "empty"), exist_ok=True)
    return out


if __name__ == "__main__":
    target = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tb-synth"
    generate(target)
    print(f"generated synthetic logs at {target}")
