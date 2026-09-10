# -*- coding: utf-8 -*-
"""参数化曲线 Hough 搜索骨架（无标注图像任务 · 题面公式即模型）

适用（SKILL.md STEP G2/G3/G4）：
  题面已给出目标形态方程、但数据**无标注**的像素级任务
  （例：2025C 问题1 『正弦状』裂隙 y = R·sin(2πx/P+β)+C，P 固定为钻孔周长）

核心机制（每条都有实测依据）：
  ① 形态方程做**主模型**：在 (R, β, C) 参数空间搜索，而非阈值+形态学（否则断裂细结构碎片化）
  ② **粗到细**搜索：粗搜定位 → 局部精搜细化（省算力 10 倍以上）
  ③ 打分用**垂直容差带内最大响应**（±tol），而非单点响应（单点对中心线偏差极敏感：
     实测同一裂隙单点得分 0.147，容差带内 0.259）
  ④ **迭代贪心 + 邻域抑制**去重：参数空间 NMS/聚类不收敛（实测近似解分散 ΔR 60px/Δβ 50°/ΔC 100px），
     必须在成像空间按实际重叠去重
  ⑤ **图内自适应阈值**：逐图归一化后响应跨图仍可差 5 倍（实测 0.331 vs 0.068），
     固定阈值会让弱响应图整张零检出 ⇒ thr = max(floor, ratio × 本图最高分)
  ⑥ **物理标定优先**：把题面物理判据（如"张开 >1mm"）换算成像素判据后再设参

用法示例（2025C 问题1）：
  PYTHONPATH= python sinusoid_hough.py --data DATA_DIR --period 244 --mm-per-px 0.3863 \
      --out OUT_DIR --all
输出：每条曲线的 (R, P, β, C) 参数表 JSON + 掩码/叠加图
"""
import argparse
import json
import os

import cv2
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True, help="图像目录（jpg/png）")
ap.add_argument("--out", required=True)
ap.add_argument("--period", type=float, required=True, help="P：形态周期（像素），通常=图像宽度（完整周期）")
ap.add_argument("--mm-per-px", type=float, default=None, help="物理标定（mm/px）；给了就输出双单位")
ap.add_argument("--topk", type=int, default=10, help="每条图像最多提取的曲线数")
ap.add_argument("--min-score", type=float, default=0.05, help="自适应阈值下限")
ap.add_argument("--thr-ratio", type=float, default=0.40, help="阈值=ratio×本图最高分")
ap.add_argument("--suppress-half", type=int, default=14, help="检出后抑制其邻域的半宽（px）")
ap.add_argument("--tol", type=int, default=4, help="打分容差带半宽（px）")
ap.add_argument("--mask-band", type=int, default=8, help="掩码扩展带半宽（px）")
ap.add_argument("--resp-ratio", type=float, default=0.62, help="掩码内响应阈值 = ratio×曲线峰值")
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)
P = float(args.period)


def feature_response(gray):
    """占位实现：光照归一化 + 线状响应。
    实际题面应按目标形态替换（本次 2025C 用双向多尺度 Frangi + 暗带加权，见
    见 CHANGELOG.md v1.3『实测数据存档』所列脚本的 response()）。"""
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE,
                          cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (61, 61))).astype(np.float32) + 1e-6
    norm = np.clip(gray.astype(np.float32) / bg, 0, 2.0)
    from skimage.filters import frangi
    fr = np.maximum(frangi(norm, sigmas=[1.3, 1.8, 2.4, 3.5], black_ridges=True),
                    frangi(norm, sigmas=[1.3, 1.8, 2.4, 3.5], black_ridges=False))
    frn = (fr / (fr.max() + 1e-9)).astype(np.float32)
    dark = np.clip(1.0 - norm / 2.0, 0, 1).astype(np.float32)
    return norm, (frn * (0.35 + 0.65 * dark)).astype(np.float32)


def curve_y(R, beta, C, x):
    return R * np.sin(2 * np.pi * x / P + beta) + C


def search_once(resp, H, W, x, tol):
    """一轮粗到细搜索：返回 (score, R, beta, C)；分数用容差带内最大响应（机制③）"""
    Rs_c = np.arange(8, min(H, 700), 8.0)
    bs_c = np.arange(0, 2 * np.pi, np.radians(10))
    Cs_c = np.arange(40, H - 40, 8.0)
    cands = []
    for C in Cs_c:
        Y = Rs_c[:, None, None] * np.sin(2 * np.pi * x[None, None, :] / P + bs_c[None, :, None]) + C
        Yi = np.clip(np.round(Y).astype(np.int32), 0, H - 1)
        sc = resp[Yi, np.broadcast_to(x[None, None, :], Yi.shape)].mean(axis=2)
        gi = np.unravel_index(int(np.argmax(sc)), sc.shape)
        cands.append((float(sc[gi]), float(Rs_c[gi[0]]), float(bs_c[gi[1]]), float(C)))
    cands.sort(reverse=True)
    best = None
    for s, R, b, C in cands[:6]:
        for R2 in np.arange(max(6, R - 10), R + 10, 2.0):
            for b2 in np.arange(b - np.radians(12), b + np.radians(12), np.radians(2)):
                for C2 in np.arange(max(20, C - 10), C + 10, 2.0):
                    yi = np.clip(np.round(curve_y(R2, b2, C2, x)).astype(np.int32), 0, H - 1)
                    s2 = float(resp[yi, x].mean())
                    if best is None or s2 > best[0]:
                        best = (s2, R2, b2, C2)
    return best


def extract(resp, H, W, x, topk, min_score, thr_ratio, suppress_half, tol):
    """迭代贪心提取 + 邻域抑制（机制④）+ 图内自适应阈值（机制⑤）"""
    work = cv2.dilate(resp, np.ones((2 * tol + 1, 1), np.float32)).copy()
    out, thr = [], None
    for _k in range(topk):
        best = search_once(work, H, W, x, tol)
        if best is None:
            break
        if thr is None:
            thr = max(min_score, thr_ratio * best[0])
        if best[0] < thr:
            break
        s, R, b, C = best
        yi = np.clip(np.round(curve_y(R, b, C, x)).astype(np.int32), 0, H - 1)
        # 边界伪影判据：近水平且贴边（按题面调整；大振幅曲线触边属正常成像，不剔除）
        touches = R < 30 and ((C - R) < 12 or (C + R) > (H - 12))
        for xi in range(W):
            y0, y1 = max(0, yi[xi] - suppress_half), min(H, yi[xi] + suppress_half + 1)
            work[y0:y1, xi] *= 0.15
        if not touches:
            out.append((s, R, b, C))
    return out, thr


def render(gray, resp, picked, x, band, resp_ratio):
    """掩码：曲线邻域内按响应自适应扩展（宽度随实际结构变化，不画等宽细线）"""
    H, W = gray.shape
    m = np.zeros((H, W), np.uint8)
    for _s, R, b, C in picked:
        y = np.clip(np.round(curve_y(R, b, C, x)).astype(np.int32), 0, H - 1)
        peak = float(resp[y, x].mean())
        t = max(peak * resp_ratio, 0.02)
        for xi in range(W):
            y0, y1 = max(0, y[xi] - band), min(H, y[xi] + band + 1)
            sel = resp[y0:y1, xi] >= t
            if sel.any():
                m[y0:y1, xi] = np.maximum(m[y0:y1, xi], sel.astype(np.uint8))
    return m


files = sorted(f for f in os.listdir(args.data) if f.lower().endswith((".jpg", ".jpeg", ".png")))
table = []
for f in files:
    gray = cv2.cvtColor(cv2.imread(os.path.join(args.data, f), cv2.IMREAD_COLOR), cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    x = np.arange(W)
    norm, resp = feature_response(gray)
    picked, thr = extract(resp, H, W, x, args.topk, args.min_score, args.thr_ratio, args.suppress_half, args.tol)
    m = render(gray, resp, picked, x, args.mask_band, args.resp_ratio)
    stem = os.path.splitext(f)[0]
    # 题目常见要求：目标像素=黑、其他=白；如需其他约定改此处
    cv2.imwrite(os.path.join(args.out, stem + "_result.png"), np.where(m > 0, 0, 255).astype(np.uint8))
    vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    vis[m > 0] = (0, 0, 255)
    cv2.imwrite(os.path.join(args.out, stem + "_vis.png"), vis)
    for k, (s, R, b, C) in enumerate(picked, 1):
        row = dict(image=stem, id=k, score=round(s, 4), R_px=round(R, 1), beta_rad=round(b, 4),
                   C_px=round(C, 1), P_px=P, beta_deg=round(np.degrees(b) % 360, 1), threshold=round(thr or 0, 4))
        if args.mm_per_px:
            row.update(R_mm=round(R * args.mm_per_px, 2), C_mm=round(C * args.mm_per_px, 2),
                       P_mm=round(P * args.mm_per_px, 2))
        table.append(row)
    print("%-16s 曲线 %d 条（阈值 %.4f）" % (stem, len(picked), thr or 0), flush=True)

with open(os.path.join(args.out, "curve_params.json"), "w", encoding="utf-8") as fh:
    json.dump(table, fh, indent=1, ensure_ascii=False)
print("共 %d 条；参数表 -> %s" % (len(table), os.path.join(args.out, "curve_params.json")))
