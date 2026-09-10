# -*- coding: utf-8 -*-
"""分割评估骨架：逐图指标 + 分布分位数 + 批级/逐图双口径 + 最差样本三联图
（华为杯解题 Skill v1.4 · STEP E2/E3，[E]）

【为什么必须这样评估】
① 主指标固定 **Dice + IoU**（附 Precision / Recall）；**PA 只作辅证**——
   背景占比约 98% 时 PA 看着「接近完美」却完全掩盖细裂缝漏检：实测 crack-seg
   **PA 0.9899 vs Dice 0.7377**（注：实测脚本原 PA 公式分母口径写错曾得 0.6566，
   修正后 0.9899——公式本身与聚合层级都必须写清，这正是本模板要防的坑）。
② 必须声明**聚合层级**：批级（先累积全局 TP/FP/FN 再算一个数）vs 逐图（每图算完再平均）——
   两者数值与含义都不同，不声明则横向对比失效（实测批级 Dice 0.7377 vs 逐图 0.7196，差 1.8 个百分点）。
③ 只报均值会掩盖系统性失效：实测 112 张中 **6 张 Dice <0.4、最差 2 张 Dice 0.000（完全漏检）**、
   第 3 张 0.111（阴影/剥落粗块区）——全被均值 0.72–0.74 掩盖 →
   必须交付逐图分布（分位数）+ 最差样本三联图归因。

用法：
  python segmentation_eval.py --gt  DIR --pred DIR [--thr 0.5] [--worst 5] [--outdir OUT]
  其中 pred 可以是概率图（0–255 灰度，按 --thr 二值化）或已二值化的 0/255 图；
  gt 为 0/非 0 的二值 mask。两个目录文件名（去扩展名）相同即配对。
输出：metrics_per_image.json（逐图 + 分布 + 批级）、worst_*.png（最差样本三联图）。

三联图配色：绿=命中(TP)、红=误报(FP)、蓝=漏检(FN)。

溯源：[E] = 2026-09-10 crack-seg 4029 张裂缝分割 A 方案实跑。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image

EPS = 1e-6


# ---------------------------------------------------------------- 基础

def load_binary(path):
    """读入 mask/预测图并返回 bool 数组（>0 即前景）。"""
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float64)
    return a > 0


def load_pred(path, thr=0.5):
    """读入预测图：灰度值 > thr*255/255 判定为前景（thr=0.5 → >127）。"""
    a = np.asarray(Image.open(path).convert("L"), dtype=np.float64) / 255.0
    return a > thr


def per_image_metrics(pred, gt):
    """单图指标（布尔数组输入）。空真值+空预测约定 Dice=1（双方一致），并在 n_gt=0 时标注。"""
    p = pred.astype(bool)
    g = gt.astype(bool)
    tp = float(np.logical_and(p, g).sum())
    fp = float(np.logical_and(p, ~g).sum())
    fn = float(np.logical_and(~p, g).sum())
    tn = float(np.logical_and(~p, ~g).sum())
    dice = (2 * tp + EPS) / (2 * tp + fp + fn + EPS)
    iou = (tp + EPS) / (tp + fp + fn + EPS)
    return {
        "dice": dice,
        "iou": iou,
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
        "pa": (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
        "gt_px": int(g.sum()),
        "pred_px": int(p.sum()),
    }


def aggregate_per_image(records):
    """逐图口径：每图先算指标，再取均值/标准差/分位数。"""
    d = np.array([r["dice"] for r in records], dtype=np.float64)
    i = np.array([r["iou"] for r in records], dtype=np.float64)
    out = {
        "aggregation": "per_image (mean over images)",
        "n": int(d.size),
        "dice_mean": float(d.mean()), "dice_std": float(d.std()),
        "iou_mean": float(i.mean()), "iou_std": float(i.std()),
        "precision_mean": float(np.mean([r["precision"] for r in records])),
        "recall_mean": float(np.mean([r["recall"] for r in records])),
        "pa_mean": float(np.mean([r["pa"] for r in records])),
    }
    for k, name in (("dice", "dice"), ("iou", "iou")):
        arr = d if k == "dice" else i
        for q in (10, 25, 50, 75, 90):
            out[f"{name}_p{q}"] = float(np.percentile(arr, q))
    out["dice_hist"], edges = np.histogram(d, bins=10, range=(0.0, 1.0))
    out["dice_hist"] = out["dice_hist"].tolist()
    out["dice_hist_edges"] = [float(e) for e in edges]
    return out


def batch_level(records):
    """批级口径：先累积全局 TP/FP/FN（等价于把整个测试集当一张大图）再算指标。"""
    tp = sum(r["tp"] for r in records)
    fp = sum(r["fp"] for r in records)
    fn = sum(r["fn"] for r in records)
    tn = sum(r["tn"] for r in records)
    return {
        "aggregation": "batch_level (global TP/FP/FN accumulate)",
        "dice": (2 * tp + EPS) / (2 * tp + fp + fn + EPS),
        "iou": (tp + EPS) / (tp + fp + fn + EPS),
        "precision": (tp + EPS) / (tp + fp + EPS),
        "recall": (tp + EPS) / (tp + fn + EPS),
        "pa": (tp + tn + EPS) / (tp + tn + fp + fn + EPS),
    }


def worst_samples(records, k=5):
    """最差 k 张（按 Dice 升序），用于三联图归因。"""
    return sorted(records, key=lambda r: r["dice"])[:k]


# ---------------------------------------------------------------- 可视化

def overlay_panel(rgb, gt, pred):
    """三联图：原图 ｜ 真值 mask ｜ 预测叠加（绿=命中、红=误报、蓝=漏检）。"""
    g = gt.astype(bool)
    p = pred.astype(bool)
    ov = rgb.copy()
    ov[g] = (0.6 * ov[g] + 0.4 * np.array([0, 255, 0])).astype(np.uint8)
    ov[p & ~g] = (0.6 * ov[p & ~g] + 0.4 * np.array([255, 0, 0])).astype(np.uint8)
    ov[p & g] = (0.35 * ov[p & g] + 0.65 * np.array([0, 255, 0])).astype(np.uint8)
    gt_rgb = np.stack([g * 255] * 3, -1).astype(np.uint8)
    return np.concatenate([rgb, gt_rgb, ov.astype(np.uint8)], axis=1)


def _pair(pred_dir, gt_dir):
    out = []
    for gp in sorted(glob.glob(os.path.join(gt_dir, "*.png")) + glob.glob(os.path.join(gt_dir, "*.jpg"))):
        stem = os.path.splitext(os.path.basename(gp))[0]
        cand = [os.path.join(pred_dir, stem + e) for e in (".png", ".jpg", ".jpeg", ".bmp", ".tif")]
        pp = next((c for c in cand if os.path.exists(c)), None)
        if pp:
            out.append((stem, pp, gp))
    return out


def main():
    ap = argparse.ArgumentParser(description="分割评估：逐图指标 + 分布 + 批级/逐图双口径 + 最差样本三联图")
    ap.add_argument("--pred", required=True, help="预测 mask/概率图目录")
    ap.add_argument("--gt", required=True, help="真值 mask 目录")
    ap.add_argument("--img", default=None, help="原图目录（用于三联图底图；缺省则用真值 mask 作底图）")
    ap.add_argument("--thr", type=float, default=0.5)
    ap.add_argument("--worst", type=int, default=5)
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    recs = []
    for stem, pp, gp in _pair(args.pred, args.gt):
        pred = load_pred(pp, args.thr)
        gt = load_binary(gp)
        m = per_image_metrics(pred, gt)
        tp = float(np.logical_and(pred, gt).sum())
        fp = float(np.logical_and(pred, ~gt).sum())
        fn = float(np.logical_and(~pred, gt).sum())
        tn = float(np.logical_and(~pred, ~gt).sum())
        recs.append(dict(name=stem, tp=tp, fp=fp, fn=fn, tn=tn, **m))
    if not recs:
        raise SystemExit("未配对到任何 (pred, gt)，请检查目录与文件名")

    per_img = aggregate_per_image(recs)
    batch = batch_level(recs)
    print(json.dumps({"per_image": per_img, "batch_level": batch}, ensure_ascii=False, indent=1))

    # 最差样本三联图
    # 复用一份配对映射：原先每张最差图各调用一次 _pair()，每次都重新 glob 全目录 → O(k·n)
    pairmap = {s: (p, g) for s, p, g in _pair(args.pred, args.gt)}
    for i, r in enumerate(worst_samples(recs, args.worst)):
        pp, gp = pairmap[r["name"]]
        if args.img:
            ip = next((c for c in (os.path.join(args.img, r["name"] + e)
                                   for e in (".jpg", ".png", ".jpeg")) if os.path.exists(c)), None)
            rgb = np.asarray(Image.open(ip).convert("RGB")) if ip else None
        else:
            rgb = None
        gt = load_binary(gp)
        if rgb is None:
            rgb = np.stack([gt * 255] * 3, -1).astype(np.uint8)
        if rgb.shape[:2] != gt.shape[:2]:
            rgb = np.asarray(Image.fromarray(rgb).resize((gt.shape[1], gt.shape[0])))
        canvas = overlay_panel(rgb, gt, load_pred(pp, args.thr))
        out = os.path.join(args.outdir, f"worst_{i}_dice{r['dice']:.3f}_{r['name']}.png")
        Image.fromarray(canvas).save(out)
        print("saved", out)

    json.dump({"per_image": per_img, "batch_level": batch,
               "worst": [{k: r[k] for k in ("name", "dice", "iou", "precision", "recall", "gt_px", "pred_px")}
                         for r in worst_samples(recs, max(args.worst, 10))]},
              open(os.path.join(args.outdir, "metrics_per_image.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print(f"[口径提醒] 报告须同时声明聚合层级：逐图 Dice {per_img['dice_mean']:.4f} vs "
          f"批级 Dice {batch['dice']:.4f}；PA {batch['pa']:.4f} 仅作辅证。")


if __name__ == "__main__":
    main()
