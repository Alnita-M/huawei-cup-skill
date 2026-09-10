# -*- coding: utf-8 -*-
"""分割标注 → 二值 mask 转换骨架（华为杯解题 Skill v1.2 · STEP E1，[E]）

用途：把几何图元标注（YOLO 归一化多边形 / COCO polygon / 折线）转成二值 mask，
      并完成『转换口径显式声明 + 一致性检查』——分割任务最容易发生口径漂移的一步。

【为什么必须声明口径】同一份多边形标注有两种主流转换口径：
  ① 多边形填充 fill=1（闭合区域填充，本骨架默认；实测 crack-seg A 方案采用此口径）
  ② 线段宽度膨胀 stroke(width=k)（细裂缝/细线场景：纯填充会低估 1–2px 细结构，
     膨胀更贴近标注意图）
两种口径可让 Dice 基准相差数个点 → 口径必须写进报告方法节，否则不可复现、不可横向对比。

用法：
  python polygon_to_mask.py --images images/test --labels labels/test [--stroke 3]
输出：正样本像素占比、单图 mask 覆盖率分位数（P10/P50/P90）、空标签图数量（数据质量告警）、
      可选把转换结果落盘为 PNG（--save DIR）。

溯源：[E] = 2026-09-10 crack-seg 4029 张裂缝分割 A 方案真实数据实跑（实测正样本占比 <1%）。
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
from PIL import Image, ImageDraw


# ---------------------------------------------------------------- 解析

def parse_yolo_polygons(label_path):
    """YOLO txt：每行 `cls x1 y1 x2 y2 ...`（坐标已归一化到 [0,1]）。

    返回 [(cls, [(x, y), ...]), ...]；行内点数 <3（不足多边形）或字段 <7 时该行丢弃。
    """
    out = []
    if not os.path.exists(label_path):
        return out
    with open(label_path, encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) < 7:
                continue
            cls = int(float(p[0]))
            c = list(map(float, p[1:]))
            pts = [(c[i], c[i + 1]) for i in range(0, len(c) - 1, 2)]
            if len(pts) >= 3:
                out.append((cls, pts))
    return out


def parse_coco_polygons(coco_json, img_id):
    """COCO polygon 标注：返回 [(category_id, [(x, y) 绝对像素坐标...]), ...]。"""
    with open(coco_json, encoding="utf-8") as f:
        coco = json.load(f)
    anns = [a for a in coco.get("annotations", []) if a.get("image_id") == img_id]
    out = []
    for a in anns:
        seg = a.get("segmentation") or []
        if isinstance(seg, list) and seg and isinstance(seg[0], list):
            flat = seg[0]
            pts = [(flat[i], flat[i + 1]) for i in range(0, len(flat) - 1, 2)]
            if len(pts) >= 3:
                out.append((a.get("category_id"), pts))
    return out


# ---------------------------------------------------------------- 转换（双口径）

def polygon_to_mask(polygons, W, H, mode="fill", stroke_width=3):
    """多边形 → 二值 mask（uint8，取值 0/1）。

    polygons : [(cls, [(x, y) 归一化坐标...]), ...]
    mode     : "fill"   闭合区域填充（默认口径）
               "stroke" 线段宽度膨胀（细结构场景）
    """
    m = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(m)
    for _cls, pts in polygons:
        xy = [(x * W, y * H) for x, y in pts]
        if len(xy) < 2:
            continue
        if mode == "fill":
            if len(xy) >= 3:
                d.polygon(xy, fill=1)
        elif mode == "stroke":
            # 画线段 + 顶点圆点（避免折角断口），宽度 k 覆盖 1–k px 细结构
            for i in range(len(xy) - 1):
                d.line([xy[i], xy[i + 1]], fill=1, width=stroke_width)
            d.line([xy[-1], xy[0]], fill=1, width=stroke_width)
            r = stroke_width / 2.0
            for x, y in xy:
                d.ellipse([x - r, y - r, x + r, y + r], fill=1)
        else:
            raise ValueError("mode 必须是 'fill' 或 'stroke'")
    return np.asarray(m, dtype=np.uint8)


# ---------------------------------------------------------------- 一致性检查

def coverage_stats(pairs, mode="fill", stroke_width=3, limit=None):
    """遍历 (image_path, label_path) 对，统计正样本占比与单图 mask 覆盖率。

    返回 dict：n_images / n_empty_labels（无标注图）/ n_empty_masks（有标注但 mask 为空，
    说明标注行格式异常或被点数过滤）/ mask 覆盖率分位数 / 正样本像素占比。
    """
    rows, empty_lbl, empty_msk = [], 0, 0
    for i, (ip, lp) in enumerate(pairs):
        if limit and i >= limit:
            break
        img = Image.open(ip)
        W, H = img.size
        polys = parse_yolo_polygons(lp)
        if not polys:
            empty_lbl += 1
            continue
        m = polygon_to_mask(polys, W, H, mode=mode, stroke_width=stroke_width)
        cov = float(m.mean())
        if cov == 0.0:
            empty_msk += 1
        rows.append(cov)
    a = np.asarray(rows, dtype=np.float64)
    pos = a * 100.0  # 单图覆盖率（%）
    return {
        "n_images": int(len(pairs)) if not limit else int(min(limit, len(pairs))),
        "n_evaluated": int(a.size),
        "n_empty_labels": int(empty_lbl),
        "n_empty_masks": int(empty_msk),
        "coverage_pct_mean": float(pos.mean()) if a.size else 0.0,
        "coverage_pct_p10": float(np.percentile(pos, 10)) if a.size else 0.0,
        "coverage_pct_p50": float(np.percentile(pos, 50)) if a.size else 0.0,
        "coverage_pct_p90": float(np.percentile(pos, 90)) if a.size else 0.0,
        "positive_pixel_ratio_pct": float(pos.mean()) if a.size else 0.0,
        "mode": mode,
        "stroke_width": stroke_width if mode == "stroke" else None,
    }


def main():
    ap = argparse.ArgumentParser(description="分割标注→mask 转换与口径一致性检查")
    ap.add_argument("--images", required=True, help="图像目录（jpg/png）")
    ap.add_argument("--labels", required=True, help="YOLO txt 标签目录（同名 .txt）")
    ap.add_argument("--mode", default="fill", choices=["fill", "stroke"])
    ap.add_argument("--stroke", type=int, default=3, help="stroke 口径的线宽（像素）")
    ap.add_argument("--limit", type=int, default=None, help="只统计前 N 张（快速试跑）")
    ap.add_argument("--save", default=None, help="把转换结果落盘为 PNG 的目录（可选）")
    args = ap.parse_args()

    imgs = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                  glob.glob(os.path.join(args.images, "*.png")))
    pairs = [(p, os.path.join(args.labels, os.path.splitext(os.path.basename(p))[0] + ".txt"))
             for p in imgs]
    st = coverage_stats(pairs, mode=args.mode, stroke_width=args.stroke, limit=args.limit)
    print(json.dumps(st, ensure_ascii=False, indent=1))

    if args.save:
        os.makedirs(args.save, exist_ok=True)
        for ip, lp in pairs[: args.limit or len(pairs)]:
            img = Image.open(ip)
            W, H = img.size
            polys = parse_yolo_polygons(lp)
            m = polygon_to_mask(polys, W, H, mode=args.mode, stroke_width=args.stroke) * 255
            Image.fromarray(m).save(os.path.join(args.save, os.path.splitext(os.path.basename(ip))[0] + ".png"))
        print(f"saved masks -> {args.save}")

    # 口径漂移告警：fill 与 stroke 差异过大时提示
    if args.mode == "fill":
        st2 = coverage_stats(pairs, mode="stroke", stroke_width=args.stroke, limit=args.limit)
        d = st2["positive_pixel_ratio_pct"] - st["positive_pixel_ratio_pct"]
        print(f"[口径对比] fill vs stroke({args.stroke}px) 正样本占比差 {d:+.3f} 个百分点；"
              f"报告方法节必须声明采用的口径。")


if __name__ == "__main__":
    main()
