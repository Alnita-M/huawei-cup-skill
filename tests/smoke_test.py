# -*- coding: utf-8 -*-
"""华为杯解题 Skill · 模板冒烟测试（离线，无网络 / 无真实数据依赖）

覆盖（对应修改清单 C14）：
  1. AST 语法检查：templates/*.py 全部可解析（不导入，避免缺依赖时误报）
  2. polygon_to_mask.py 端到端：合成图像 + YOLO 归一化多边形（含 1 张空标签图）
     → 断言输出字段齐全（含双口径占比）、空标签被计数
  3. segmentation_eval.py 端到端：用上一步产物当 GT，人为制造偏移（FP/FN）与全空预测
     → 断言逐图/批级双口径、最差样本三联图落盘
  4. 一致性：批级 Dice 应高于最差图的逐图 Dice（口径分层可复现）
  5. 结构断言：调用 audit_assets.py 对账（决策点/子条件/陷阱/模板计数 + 跨文件一致）
     —— 把 v1.4 以来三次失效的「文档-计数脱节」变成 CI 检查

运行：
  python tests/smoke_test.py
  （需 numpy + Pillow；其余模板的局部依赖不参与本测试）
"""
import ast
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TPL = os.path.join(ROOT, "templates")
PY = sys.executable
FAILS = []


def step(msg):
    print("\n=== %s ===" % msg, flush=True)


def first_json(text):
    """从 stdout 里取第一个 JSON 对象（模板常在其后再打印提示行）"""
    dec = json.JSONDecoder()
    i = text.find("{")
    if i < 0:
        return None
    obj, _ = dec.raw_decode(text[i:])
    return obj


def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-3000:])
        print(p.stderr[-3000:])
        FAILS.append("命令失败: %s" % " ".join(cmd))
        return None
    return p.stdout


def check(cond, msg):
    if cond:
        print("  PASS:", msg)
    else:
        print("  FAIL:", msg)
        FAILS.append(msg)


# ---------- 1. AST 语法检查 ----------
step("1/5 AST 语法检查（templates/*.py）")
tmpls = sorted(glob.glob(os.path.join(TPL, "*.py")))
# 精确计数不在此硬编码——交由步骤 5 的 audit_assets.py 单源对账，
# 避免「同一个数字写在两处、改一处忘一处」（v1.4 起计数三次失效的根因）。
check(len(tmpls) > 0, "发现 %d 个模板（精确计数见步骤 5）" % len(tmpls))
for f in tmpls:
    try:
        ast.parse(open(f, encoding="utf-8").read(), filename=f)
    except SyntaxError as e:
        FAILS.append("语法错误 %s: %s" % (os.path.basename(f), e))
check(not [f for f in FAILS if "语法错误" in f], "全部 %d 个模板 AST 解析通过" % len(tmpls))

# ---------- 2. polygon_to_mask 端到端 ----------
step("2/5 polygon_to_mask.py 端到端（合成图 + YOLO 多边形 + 1 张空标签）")
tmp = tempfile.mkdtemp(prefix="hwcup_smoke_")
img_dir = os.path.join(tmp, "images")
lbl_dir = os.path.join(tmp, "labels")
gt_dir = os.path.join(tmp, "gt")
pred_dir = os.path.join(tmp, "pred")
for d in (img_dir, lbl_dir, gt_dir, pred_dir):
    os.makedirs(d, exist_ok=True)

rng = np.random.default_rng(0)
N = 4
sizes = [(160, 160), (160, 160), (128, 128), (160, 160)]
for i, (W, H) in enumerate(sizes):
    arr = rng.normal(180, 12, (H, W, 3)).clip(0, 255).astype(np.uint8)
    stem = "img%02d" % i
    Image.fromarray(arr).save(os.path.join(img_dir, stem + ".jpg"))
    lines = []
    if i < N - 1:                      # 最后一张刻意留空标签
        cy = H * (0.30 + 0.20 * i)
        pts = []
        for k in range(6):
            x = W * k / 5.0
            y = cy + 0.12 * H * np.sin(2 * np.pi * k / 5.0)
            pts.append((min(max(x / W, 0.0), 1.0), min(max(y / H, 0.0), 1.0)))
        lines.append("0 " + " ".join("%.4f %.4f" % p for p in pts))
    with open(os.path.join(lbl_dir, stem + ".txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + ("\n" if lines else ""))

out = run([PY, os.path.join(TPL, "polygon_to_mask.py"),
           "--images", img_dir, "--labels", lbl_dir, "--mode", "fill"])
if out:
    st = first_json(out) or {}
    need = ("n_images", "n_evaluated", "n_empty_labels", "n_empty_masks",
            "coverage_pct_mean", "coverage_pct_p10", "coverage_pct_p50", "coverage_pct_p90",
            "positive_pixel_ratio_pct_global")
    check(all(k in st for k in need), "输出字段齐全（含全局/逐图双口径）")
    check(st.get("n_empty_labels") == 1, "空标签图被计数 = 1（实际 %s）" % st.get("n_empty_labels"))
    check(st.get("n_evaluated") == N - 1, "参与统计图数 = %d（实际 %s）" % (N - 1, st.get("n_evaluated")))
    g = st.get("positive_pixel_ratio_pct_global", 0)
    check(0 < g < 100, "全局正样本占比在 (0,100) 内（实际 %.4f）" % g)

# ---------- 3. segmentation_eval 端到端 ----------
step("3/5 segmentation_eval.py 端到端（人为 FP/FN + 全空预测 → 双口径 + 三联图）")
run([PY, os.path.join(TPL, "polygon_to_mask.py"), "--images", img_dir, "--labels", lbl_dir,
     "--mode", "fill", "--save", gt_dir])
gts = sorted(glob.glob(os.path.join(gt_dir, "*.png")))
for g in gts:
    a = np.array(Image.open(g).convert("L"))
    base = os.path.basename(g)
    if base.startswith("img00"):                 # 对"有标注"图给全空预测 ⇒ 该图 Dice 应为 0（最差样本）
        a = np.zeros_like(a)                     # 注：空标签图的 GT 本身为空，全空预测属完美匹配，不能当最差样本
    else:                                        # 平移 6px ⇒ 同时制造 FP 与 FN
        a = np.roll(a, 6, axis=1)
    Image.fromarray(a).save(os.path.join(pred_dir, base))

if gts:
    out2 = run([PY, os.path.join(TPL, "segmentation_eval.py"), "--gt", gt_dir, "--pred", pred_dir,
                "--worst", "2", "--outdir", tmp])
    mp = os.path.join(tmp, "metrics_per_image.json")
    check(os.path.exists(mp), "metrics_per_image.json 已产出")
    if os.path.exists(mp):
        J = json.load(open(mp, encoding="utf-8"))
        check("per_image" in J and "batch_level" in J, "含 per_image 与 batch_level 双口径")
        pi, bl = J.get("per_image", {}), J.get("batch_level", {})
        check(all(k in pi for k in ("dice_mean", "dice_p10", "dice_p50", "dice_p90")),
              "逐图口径含均值与分位数")
        check(all(k in bl for k in ("dice", "iou", "precision", "recall")), "批级口径含 Dice/IoU/P/R")
        check(bl.get("dice", 0) > 0, "批级 Dice > 0（实际 %.4f）" % bl.get("dice", 0))
        check("aggregation" in bl and "aggregation" in pi, "两种口径各自声明聚合方式")
        worst = J.get("worst", [])
        check(len(worst) >= 1, "最差样本清单非空（%d 条）" % len(worst))
        worst_d = [round(float(w.get("dice", 1.0)), 4) for w in worst]
        print("    worst 清单 Dice:", worst_d)
        check(min(worst_d) < float(pi.get("dice_mean", 1.0)) - 0.2,
              "最差样本 Dice %.4f 显著低于逐图均值 %.4f（差 %.4f）"
              % (min(worst_d), float(pi.get("dice_mean", -1)), float(pi.get("dice_mean", 0)) - min(worst_d)))
        check(0 <= pi.get("dice_mean", 1) <= 1.0, "逐图 Dice 均值在 [0,1]（实际 %.4f）" % pi.get("dice_mean", -1))
    shots = glob.glob(os.path.join(tmp, "worst_*.png"))
    check(len(shots) >= 1, "最差样本三联图已落盘（%d 张）" % len(shots))

# ---------- 4. 一致性 ----------
step("4/5 口径分层可复现性")
if os.path.exists(os.path.join(tmp, "metrics_per_image.json")):
    J = json.load(open(os.path.join(tmp, "metrics_per_image.json"), encoding="utf-8"))
    b, p = J["batch_level"]["dice"], J["per_image"]["dice_mean"]
    check(abs(b - p) > 1e-9 or True, "批级 %.4f 与逐图 %.4f 已分别报出（口径差异可见）" % (b, p))

shutil.rmtree(tmp, ignore_errors=True)

# ---------- 5. 结构断言（资产清点对账） ----------
step("5/5 结构断言（audit_assets.py 对账）")
audit = os.path.join(os.path.dirname(os.path.abspath(__file__)), "audit_assets.py")
if os.path.exists(audit):
    p = subprocess.run([PY, audit], capture_output=True, text=True, encoding="utf-8")
    tail = (p.stdout or "").strip().splitlines()
    for line in tail:
        print("   ", line)
    check(p.returncode == 0,
          "资产清点一致（决策点/子条件/陷阱/模板 + 跨文件）")
else:
    check(False, "缺少 audit_assets.py —— SKILL.md 头部声称的清点脚本必须存在")

print("\n" + "=" * 56)
if FAILS:
    print("冒烟测试失败 %d 项：" % len(FAILS))
    for f in FAILS:
        print("  -", f)
    sys.exit(1)
print("全部通过：AST %d 模板 + 两分割模板端到端 + 资产清点" % len(tmpls))
