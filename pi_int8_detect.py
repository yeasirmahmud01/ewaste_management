#!/usr/bin/env python3
"""
================================================================================
  Real-Time INT8 ONNX Object Detection on Raspberry Pi
  ---------------------------------------------------------------------------
  Implements the full pipeline from Nugroho et al. (2025):
    "Automated Component Detection for Quality PCB Using YOLO Algorithm
     with IoT Real-Time Streaming on Raspberry Pi"

  Features:
    - INT8-quantized YOLO11-nano (distilled) via ONNX Runtime (CPU only)
    - Per-stage latency timing: capture / preprocess / inference / postprocess / draw
    - Rolling-window smoothed FPS + HUD overlay
    - End-of-run summary table (avg / p50 / p95 latency, throughput)
    - Optional MQTT publish to Node-RED / InfluxDB (IoT section of paper)
    - Optional CSV log of every frame
    - Works headless (SSH) or with X display
    - Sources: USB webcam, video file, or image folder

  Tested on: Raspberry Pi 4B (4 GB), Raspberry Pi OS 64-bit, Python 3.9+

INSTALL:
    sudo apt update && sudo apt install -y python3-opencv libatlas-base-dev
    pip install onnxruntime numpy pyyaml paho-mqtt

RUN (live camera, headless):
    python3 pi_int8_detect.py \
        --model yolo_nano_distilled_int8.onnx \
        --config int8_eval_config.yml \
        --source 0 --headless --csv latency_log.csv

RUN (with MQTT / Node-RED):
    python3 pi_int8_detect.py --model yolo_nano_distilled_int8.onnx \
        --source 0 --mqtt-broker 192.168.1.100 --mqtt-topic pcb/detections

RUN (video file):
    python3 pi_int8_detect.py --model yolo_nano_distilled_int8.onnx --source pcb.mp4

RUN (image folder benchmark):
    python3 pi_int8_detect.py --model yolo_nano_distilled_int8.onnx --source ./test_imgs
================================================================================
"""

import argparse
import csv
import json
import math
import os
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

# ── optional deps ──────────────────────────────────────────────────────────────
try:
    import onnxruntime as ort
except ImportError:
    sys.exit("[ERROR] onnxruntime not installed.  Run: pip install onnxruntime")

try:
    import yaml
except ImportError:
    yaml = None

try:
    import paho.mqtt.client as mqtt
    MQTT_AVAILABLE = True
except ImportError:
    MQTT_AVAILABLE = False


# ==============================================================================
# 1.  PRE / POST-PROCESSING
#     Byte-identical to notebook Block 15C_EVAL so Pi results match validation.
# ==============================================================================

def letterbox(img, target=640, stride=32, color=(114, 114, 114)):
    """Resize + pad to a square multiple of `stride`."""
    h0, w0 = img.shape[:2]
    scale = min(target / h0, target / w0)
    nh = math.ceil(int(round(h0 * scale)) / stride) * stride
    nw = math.ceil(int(round(w0 * scale)) / stride) * stride

    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target, target, 3), color, dtype=np.uint8)
    pad_top  = (target - nh) // 2
    pad_left = (target - nw) // 2
    canvas[pad_top:pad_top + nh, pad_left:pad_left + nw] = img_resized
    return canvas, scale, (pad_top, pad_left)


def preprocess(bgr_img, imgsz=640):
    """BGR HWC uint8  →  float32 NCHW [0, 1]"""
    lbx, scale, pads = letterbox(bgr_img, target=imgsz)
    rgb    = cv2.cvtColor(lbx, cv2.COLOR_BGR2RGB)
    tensor = (rgb.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
    return np.ascontiguousarray(tensor), scale, pads


def undo_letterbox(boxes_xyxy, scale, pad_top, pad_left, orig_h, orig_w):
    if boxes_xyxy.shape[0] == 0:
        return boxes_xyxy
    b = boxes_xyxy.copy().astype(np.float32)
    b[:, [0, 2]] -= pad_left
    b[:, [1, 3]] -= pad_top
    b /= scale
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0, orig_w)
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0, orig_h)
    return b


def decode_yolo_output(output):
    """
    output shape: [1, 4+nc, num_anchors]
    Class scores are already sigmoided (export=True during training).
    Returns: boxes_xyxy, scores, class_ids
    """
    pred          = output[0].T                    # (num_anchors, 4+nc)
    boxes_cxcywh  = pred[:, :4]
    class_scores  = pred[:, 4:]

    x1 = boxes_cxcywh[:, 0] - boxes_cxcywh[:, 2] / 2
    y1 = boxes_cxcywh[:, 1] - boxes_cxcywh[:, 3] / 2
    x2 = boxes_cxcywh[:, 0] + boxes_cxcywh[:, 2] / 2
    y2 = boxes_cxcywh[:, 1] + boxes_cxcywh[:, 3] / 2
    boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)

    class_ids = np.argmax(class_scores, axis=1).astype(np.int32)
    scores    = class_scores[np.arange(len(class_ids)), class_ids]
    return boxes_xyxy, scores, class_ids


def nms_cpu(boxes_xyxy, scores, iou_thr=0.45):
    """Class-agnostic NMS in pure NumPy (no extra deps on Pi)."""
    if len(boxes_xyxy) == 0:
        return np.array([], dtype=np.int32)

    x1, y1, x2, y2 = (boxes_xyxy[:, i] for i in range(4))
    areas = (x2 - x1).clip(0) * (y2 - y1).clip(0)
    order = scores.argsort()[::-1]
    keep  = []

    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        ix1  = np.maximum(x1[i], x1[rest])
        iy1  = np.maximum(y1[i], y1[rest])
        ix2  = np.minimum(x2[i], x2[rest])
        iy2  = np.minimum(y2[i], y2[rest])
        inter = (ix2 - ix1).clip(0) * (iy2 - iy1).clip(0)
        union = areas[i] + areas[rest] - inter + 1e-9
        order = rest[inter / union <= iou_thr]

    return np.array(keep, dtype=np.int32)


# ==============================================================================
# 2.  VISUALISATION
# ==============================================================================

def _color_for(cls_id):
    rng = np.random.default_rng(cls_id * 9173 + 1)
    return tuple(int(c) for c in rng.integers(64, 255, size=3))


def draw_detections(img, dets, class_names):
    """Draw bounding boxes + labels.  Returns per-class counts dict."""
    counts = {}
    for d in dets:
        x1, y1, x2, y2 = [int(v) for v in d["box"]]
        cid  = d["cls_id"]
        conf = d["conf"]
        name = class_names[cid] if cid < len(class_names) else str(cid)
        col  = _color_for(cid)

        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        label = f"{name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), col, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        counts[name] = counts.get(name, 0) + 1
    return counts


def draw_hud(img, fps, lat_ms, stage_ms, counts):
    """Heads-up display: FPS, per-stage latency, detection counts."""
    h, w = img.shape[:2]
    pad  = 8
    lines = [
        f"FPS: {fps:5.2f}     end-to-end: {lat_ms:6.1f} ms",
        f"pre {stage_ms['pre']:5.1f}  inf {stage_ms['inf']:5.1f}  "
        f"post {stage_ms['post']:5.1f}  draw {stage_ms['draw']:5.1f}",
    ]
    if counts:
        top = sorted(counts.items(), key=lambda x: -x[1])[:5]
        lines.append("  ".join(f"{k}:{v}" for k, v in top))

    box_h = 18 * len(lines) + pad
    cv2.rectangle(img, (0, 0), (w, box_h), (0, 0, 0), -1)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, 16 + i * 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1,
                    cv2.LINE_AA)


# ==============================================================================
# 3.  SESSION + CONFIG LOADING
# ==============================================================================

def load_class_names(config_path, classes_txt, n_classes_fallback=80):
    """Priority: YAML config → classes.txt → generic 'cls_N' labels."""
    if config_path and Path(config_path).exists():
        if yaml is None:
            print("[warn] pyyaml not installed – skipping config.yml")
        else:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            names = cfg.get("dataset", {}).get("class_names")
            if names:
                return list(names), cfg
    if classes_txt and Path(classes_txt).exists():
        with open(classes_txt) as f:
            return [l.strip() for l in f if l.strip()], None
    return [f"cls_{i}" for i in range(n_classes_fallback)], None


def build_session(model_path, num_threads):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    return ort.InferenceSession(
        model_path, sess_options=so,
        providers=["CPUExecutionProvider"]
    )


# ==============================================================================
# 4.  IoT: MQTT PUBLISHER  (mirrors Node-RED integration in the paper)
# ==============================================================================

class MQTTPublisher:
    """Publishes detection results to a broker (Node-RED / InfluxDB gateway)."""

    def __init__(self, broker, port=1883, topic="pcb/detections"):
        if not MQTT_AVAILABLE:
            raise RuntimeError("paho-mqtt not installed. Run: pip install paho-mqtt")
        self.topic  = topic
        self.client = mqtt.Client()
        self.client.connect(broker, port, keepalive=60)
        self.client.loop_start()
        print(f"[MQTT] connected to {broker}:{port}  topic={topic}")

    def publish(self, frame_idx, counts, fps, lat_ms):
        payload = json.dumps({
            "frame":   frame_idx,
            "fps":     round(fps, 2),
            "lat_ms":  round(lat_ms, 1),
            "counts":  counts,
        })
        self.client.publish(self.topic, payload)

    def stop(self):
        self.client.loop_stop()
        self.client.disconnect()


# ==============================================================================
# 5.  VIDEO / IMAGE SOURCES
# ==============================================================================

class FrameSource:
    """Unified iterator: USB camera / video file / image folder."""

    def __init__(self, source, cam_w=640, cam_h=480, cam_fps=30):
        self.kind   = None
        self.cap    = None
        self.images = []
        self.idx    = 0

        if isinstance(source, str) and Path(source).is_dir():
            self.kind   = "folder"
            exts        = {".jpg", ".jpeg", ".png", ".bmp"}
            self.images = sorted(
                p for p in Path(source).iterdir() if p.suffix.lower() in exts
            )
            if not self.images:
                raise RuntimeError(f"No images found in folder: {source}")
        else:
            src = int(source) if str(source).isdigit() else source
            self.cap = cv2.VideoCapture(src)
            if not self.cap.isOpened():
                raise RuntimeError(f"Cannot open source: {source}")
            self.kind = "video"
            if isinstance(src, int):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  cam_w)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam_h)
                self.cap.set(cv2.CAP_PROP_FPS,          cam_fps)

    def read(self):
        if self.kind == "folder":
            if self.idx >= len(self.images):
                return None, True
            img = cv2.imread(str(self.images[self.idx]))
            self.idx += 1
            return img, self.idx >= len(self.images)
        ok, frame = self.cap.read()
        return (frame if ok else None), (not ok)

    def release(self):
        if self.cap is not None:
            self.cap.release()


# ==============================================================================
# 6.  SUMMARY TABLE  (paper-style, printed at end of run)
# ==============================================================================

def print_summary(frame_idx, run_secs, inf_times, e2e_times,
                  pre_times, post_times, draw_times, total_dets):
    if frame_idx == 0:
        return
    inf_arr  = np.array(inf_times)
    e2e_arr  = np.array(e2e_times)
    pre_arr  = np.array(pre_times)
    post_arr = np.array(post_times)
    draw_arr = np.array(draw_times)

    def row(label, arr):
        return (f"  {label:<22} avg {arr.mean():6.1f} ms | "
                f"p50 {np.percentile(arr,50):6.1f} ms | "
                f"p95 {np.percentile(arr,95):6.1f} ms | "
                f"min {arr.min():5.1f}  max {arr.max():.1f}")

    print("\n" + "=" * 72)
    print("  RUN SUMMARY  –  matching Table 1 / Section 4 of Nugroho et al.")
    print("=" * 72)
    print(f"  Frames processed    : {frame_idx}")
    print(f"  Wall time           : {run_secs:.2f} s")
    print(f"  Avg throughput      : {frame_idx / run_secs:.2f} FPS")
    print("-" * 72)
    print(row("Preprocess",  pre_arr))
    print(row("Inference",   inf_arr))
    print(row("Postprocess", post_arr))
    print(row("Draw",        draw_arr))
    print(row("End-to-end",  e2e_arr))
    print("-" * 72)
    print(f"  Total detections    : {total_dets}  "
          f"(avg {total_dets / frame_idx:.2f} / frame)")
    print("=" * 72)


# ==============================================================================
# 7.  MAIN LOOP
# ==============================================================================

def main():
    ap = argparse.ArgumentParser(
        description="INT8 ONNX YOLO detection on Raspberry Pi "
                    "(Nugroho et al. 2025 pipeline)")

    # ── model / source ────────────────────────────────────────────────────────
    ap.add_argument("--model",       required=True,
                    help="Path to INT8 .onnx model file")
    ap.add_argument("--source",      default="0",
                    help="Camera index (0), video file, or image folder")
    ap.add_argument("--config",      default="int8_eval_config.yml",
                    help="YAML config (class names + thresholds)")
    ap.add_argument("--classes-txt", default="classes.txt",
                    help="Fallback: one class name per line")

    # ── inference settings ────────────────────────────────────────────────────
    ap.add_argument("--imgsz",   type=int,   default=640)
    ap.add_argument("--conf",    type=float, default=0.25,
                    help="Confidence threshold")
    ap.add_argument("--iou",     type=float, default=0.45,
                    help="NMS IoU threshold")
    ap.add_argument("--threads", type=int,   default=4,
                    help="ORT intra-op threads (4 recommended for Pi 4B/5)")
    ap.add_argument("--warmup",  type=int,   default=5,
                    help="Dummy inference passes before timing starts")

    # ── camera ────────────────────────────────────────────────────────────────
    ap.add_argument("--cam-width",  type=int, default=640)
    ap.add_argument("--cam-height", type=int, default=480)
    ap.add_argument("--cam-fps",    type=int, default=30)

    # ── output / logging ──────────────────────────────────────────────────────
    ap.add_argument("--headless",   action="store_true",
                    help="No display window (use over SSH)")
    ap.add_argument("--save-video", default="",
                    help="Save annotated output to file (e.g. out.mp4)")
    ap.add_argument("--csv",        default="",
                    help="CSV path for per-frame latency log")
    ap.add_argument("--max-frames", type=int, default=0,
                    help="Stop after N frames (0 = run until quit/EOF)")

    # ── IoT / MQTT (Section 3.2.2 of paper) ──────────────────────────────────
    ap.add_argument("--mqtt-broker", default="",
                    help="MQTT broker IP (e.g. 192.168.1.100).  "
                         "Leave blank to disable.")
    ap.add_argument("--mqtt-port",   type=int, default=1883)
    ap.add_argument("--mqtt-topic",  default="pcb/detections")
    ap.add_argument("--mqtt-every",  type=int, default=1,
                    help="Publish every N frames (reduce bandwidth)")

    args = ap.parse_args()

    # ── load class names & override thresholds from YAML if present ──────────
    class_names, cfg = load_class_names(
        args.config, args.classes_txt, n_classes_fallback=80)
    if cfg is not None:
        ev = cfg.get("evaluation", {})
        args.conf  = ev.get("conf_threshold",   args.conf)
        args.iou   = ev.get("nms_iou_threshold", args.iou)
        args.imgsz = ev.get("imgsz",             args.imgsz)

    print("=" * 72)
    print("  Raspberry Pi INT8 ONNX Object Detection")
    print("  Following: Nugroho et al., Jurnal Infotel 17(2), 2025")
    print("=" * 72)
    print(f"  Model       : {args.model}")
    print(f"  Classes     : {len(class_names)}  → {class_names[:5]}")
    print(f"  imgsz       : {args.imgsz}   conf={args.conf}   iou={args.iou}")
    print(f"  Source      : {args.source}")
    print(f"  Threads     : {args.threads}")

    # ── build ORT session ─────────────────────────────────────────────────────
    sess     = build_session(args.model, args.threads)
    inp_name = sess.get_inputs()[0].name
    print(f"  ORT input   : {inp_name}  {sess.get_inputs()[0].shape}")
    print(f"  Providers   : {sess.get_providers()}")

    # ── warmup (first inference on Pi is 3-5× slower than steady-state) ──────
    print(f"\n[warmup] running {args.warmup} dummy passes …")
    dummy = np.zeros((1, 3, args.imgsz, args.imgsz), dtype=np.float32)
    for _ in range(args.warmup):
        sess.run(None, {inp_name: dummy})
    print("[warmup] done\n")

    # ── open source ───────────────────────────────────────────────────────────
    src = FrameSource(args.source, args.cam_width, args.cam_height, args.cam_fps)

    # ── optional MQTT publisher ───────────────────────────────────────────────
    mqtt_pub = None
    if args.mqtt_broker:
        try:
            mqtt_pub = MQTTPublisher(args.mqtt_broker, args.mqtt_port,
                                     args.mqtt_topic)
        except Exception as e:
            print(f"[warn] MQTT init failed: {e}  – continuing without IoT")

    # ── optional CSV logger ───────────────────────────────────────────────────
    csv_f, csv_w = None, None
    if args.csv:
        csv_f = open(args.csv, "w", newline="")
        csv_w = csv.writer(csv_f)
        csv_w.writerow(["frame", "pre_ms", "inf_ms", "post_ms",
                         "draw_ms", "total_ms", "fps", "n_dets"])

    # ── optional video writer ─────────────────────────────────────────────────
    writer = None

    # ── rolling windows for stable HUD readings (30-frame window) ────────────
    WIN = 30
    fps_win  = deque(maxlen=WIN)
    pre_win  = deque(maxlen=WIN)
    inf_win  = deque(maxlen=WIN)
    post_win = deque(maxlen=WIN)
    draw_win = deque(maxlen=WIN)

    # ── accumulators for end-of-run summary ───────────────────────────────────
    all_inf   = []
    all_e2e   = []
    all_pre   = []
    all_post  = []
    all_draw  = []
    total_dets  = 0
    frame_idx   = 0
    t_run_start = time.perf_counter()

    hint = "press 'q' to quit" if not args.headless else "Ctrl-C to stop"
    print(f"[running]  {hint}\n")

    try:
        while True:
            # ── 1. CAPTURE ────────────────────────────────────────────────────
            t0 = time.perf_counter()
            frame, last = src.read()
            if frame is None:
                break
            t_cap = time.perf_counter()

            # ── 2. PREPROCESS ─────────────────────────────────────────────────
            tensor, scale, (pad_top, pad_left) = preprocess(frame, args.imgsz)
            t_pre = time.perf_counter()

            # ── 3. INFERENCE ──────────────────────────────────────────────────
            raw = sess.run(None, {inp_name: tensor})
            t_inf = time.perf_counter()

            # ── 4. POSTPROCESS: decode → filter → NMS → unpad ────────────────
            boxes, scores, cids = decode_yolo_output(raw[0])
            mask = scores >= args.conf
            boxes, scores, cids = boxes[mask], scores[mask], cids[mask]

            if len(boxes):
                keep  = nms_cpu(boxes, scores, iou_thr=args.iou)
                boxes = undo_letterbox(
                    boxes[keep], scale, pad_top, pad_left,
                    frame.shape[0], frame.shape[1]
                )
                scores, cids = scores[keep], cids[keep]
            else:
                boxes = np.empty((0, 4), np.float32)

            dets = [
                {"cls_id": int(c), "conf": float(s), "box": boxes[i].tolist()}
                for i, (c, s) in enumerate(zip(cids, scores))
            ]
            t_post = time.perf_counter()

            # ── 5. DRAW ───────────────────────────────────────────────────────
            counts  = draw_detections(frame, dets, class_names)

            pre_ms   = (t_pre  - t_cap)  * 1000
            inf_ms   = (t_inf  - t_pre)  * 1000
            post_ms  = (t_post - t_inf)  * 1000
            total_ms = (t_post - t_cap)  * 1000      # capture excluded from e2e

            fps_win.append(1000.0 / total_ms if total_ms > 0 else 0)
            pre_win.append(pre_ms)
            inf_win.append(inf_ms)
            post_win.append(post_ms)

            smooth_fps = sum(fps_win) / len(fps_win)
            smooth_stage = {
                "pre":  sum(pre_win)  / len(pre_win),
                "inf":  sum(inf_win)  / len(inf_win),
                "post": sum(post_win) / len(post_win),
                "draw": sum(draw_win) / len(draw_win) if draw_win else 0.0,
            }
            draw_hud(frame, smooth_fps, total_ms, smooth_stage, counts)

            t_draw  = time.perf_counter()
            draw_ms = (t_draw - t_post) * 1000
            draw_win.append(draw_ms)

            # ── accumulate ────────────────────────────────────────────────────
            all_pre.append(pre_ms);   all_inf.append(inf_ms)
            all_post.append(post_ms); all_draw.append(draw_ms)
            all_e2e.append(total_ms)
            total_dets += len(dets)
            frame_idx  += 1

            # ── CSV log ───────────────────────────────────────────────────────
            if csv_w:
                csv_w.writerow([frame_idx,
                                 f"{pre_ms:.3f}",  f"{inf_ms:.3f}",
                                 f"{post_ms:.3f}", f"{draw_ms:.3f}",
                                 f"{total_ms:.3f}", f"{smooth_fps:.3f}",
                                 len(dets)])

            # ── MQTT publish (IoT streaming to Node-RED) ──────────────────────
            if mqtt_pub and frame_idx % args.mqtt_every == 0:
                try:
                    mqtt_pub.publish(frame_idx, counts, smooth_fps, total_ms)
                except Exception:
                    pass

            # ── video writer ──────────────────────────────────────────────────
            if writer is None and args.save_video:
                h, w   = frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.save_video, fourcc, 15.0, (w, h))
            if writer:
                writer.write(frame)

            # ── display ───────────────────────────────────────────────────────
            if not args.headless:
                cv2.imshow("INT8 YOLO11n – Pi", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames and frame_idx >= args.max_frames:
                break
            if last and src.kind == "folder":
                break

    except KeyboardInterrupt:
        print("\n[interrupted by user]")
    finally:
        run_secs = time.perf_counter() - t_run_start
        src.release()
        if writer:
            writer.release()
        if csv_f:
            csv_f.close()
        if mqtt_pub:
            mqtt_pub.stop()
        if not args.headless:
            cv2.destroyAllWindows()

        # ── paper-style summary table ──────────────────────────────────────
        print_summary(frame_idx, run_secs,
                      all_inf, all_e2e, all_pre, all_post, all_draw,
                      total_dets)


if __name__ == "__main__":
    main()
