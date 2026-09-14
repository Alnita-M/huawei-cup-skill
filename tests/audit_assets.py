# -*- coding: utf-8 -*-
"""华为杯解题 Skill · 资产清点脚本（落实 SKILL.md 头部『计数纪律』）

背景：SKILL.md 头部曾声称「以清点脚本输出为唯一事实源」，但仓库内并无此脚本——
声明是愿景不是机制，计数对账仍靠人肉，这正是『文档-计数脱节』在 v1.4 / v1.5 / v1.5.2
三次失效的共同根源。本脚本即为那句话补上实体。

检查项（任一不符 → 非零退出码，可直接挂 CI 或提交前自检）：
  1. SKILL.md 编号决策点计数（A/B/C/D/E/F/G/M/P/R/S/U/V 前缀 + 【 格式）
  2. SKILL.md 子条件计数（行首 `- 条件：`）
  3. 陷阱库.md 条目 T1..TN 连续无缺口
  4. templates/*.py 文件数
  5. 交叉一致性：SKILL.md 头部『资产清点』段 与 README.md 表格 的数字必须相等，
     且都等于上列实测值

用法：
    python tests/audit_assets.py            # 人读输出
    python tests/audit_assets.py --quiet    # 只报结论（供 smoke_test 调用）
退出码：0 = 全部一致；1 = 有账目不符；2 = 文件缺失或无法解析
"""
import argparse
import glob
import os
import re
import sys

# 控制台编码兜底：Windows 默认代码页下中文会乱码，导致人读输出不可用
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL = os.path.join(ROOT, "SKILL.md")
TRAPS = os.path.join(ROOT, "陷阱库.md")
README = os.path.join(ROOT, "README.md")
TPL_DIR = os.path.join(ROOT, "templates")

# 决策点前缀（与 SKILL.md 第三节实际使用的编号体系一致）
PREFIXES = ("A", "B", "C", "D", "E", "F", "G", "M", "P", "R", "S", "U", "V")

FAILS = []
REPORT = []


def _read(path):
    if not os.path.exists(path):
        print("错误：缺少文件 %s" % path, file=sys.stderr)
        sys.exit(2)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


def count_decision_points(text):
    """统计形如 `A1【...】`、`M13【...】` 的编号决策点。"""
    hits = re.findall(r"(?m)^([A-Z]{1,2})(\d+)【", text)
    per = {}
    total = 0
    for pre, _num in hits:
        if pre in PREFIXES:
            per[pre] = per.get(pre, 0) + 1
            total += 1
    return total, per


def count_subconditions(text):
    """统计行首 `- 条件：` 形式的子条件（A1/M1/F1 等多分支决策点内部）。"""
    return len(re.findall(r"(?m)^- 条件：", text))


def count_traps(text):
    """统计陷阱库条目标题 `**T54 ...**`，并检查编号连续性。"""
    nums = sorted({int(n) for n in re.findall(r"(?m)^\*\*T(\d+)", text)})
    return nums


def parse_skill_header_counts(text):
    """从 SKILL.md 头部『资产清点』段取声明值。"""
    out = {}
    m = re.search(r"编号决策点\s*\*\*(\d+)\*\*\s*个", text)
    if m:
        out["决策点"] = int(m.group(1))
    m = re.search(r"子条件\s*\*\*(\d+)\*\*\s*条", text)
    if m:
        out["子条件"] = int(m.group(1))
    m = re.search(r"陷阱\s*\*\*(\d+)\*\*\s*条", text)
    if m:
        out["陷阱"] = int(m.group(1))
    m = re.search(r"代码模板\s*\*\*(\d+)\*\*\s*个", text)
    if m:
        out["模板"] = int(m.group(1))
    return out


def parse_readme_counts(text):
    """从 README.md 内容结构表取声明值。"""
    out = {}
    m = re.search(r"\*\*(\d+)\s*个编号决策点", text)
    if m:
        out["决策点"] = int(m.group(1))
    m = re.search(r"(\d+)\s*条子条件", text)
    if m:
        out["子条件"] = int(m.group(1))
    m = re.search(r"\*\*(\d+)\s*条\*\*已证陷阱", text)
    if m:
        out["陷阱"] = int(m.group(1))
    m = re.search(r"\*\*(\d+)\s*个\*\*\s*Python\s*函数骨架", text)
    if m:
        out["模板"] = int(m.group(1))
    return out


def check(cond, msg):
    REPORT.append(("PASS" if cond else "FAIL", msg))
    if not cond:
        FAILS.append(msg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true", help="只输出结论行")
    args = ap.parse_args()

    skill_txt = _read(SKILL)
    trap_txt = _read(TRAPS)
    readme_txt = _read(README)

    # ---------- 实测值 ----------
    dp_total, dp_per = count_decision_points(skill_txt)
    sub_total = count_subconditions(skill_txt)
    trap_nums = count_traps(trap_txt)
    tpl_files = sorted(glob.glob(os.path.join(TPL_DIR, "*.py")))

    if not args.quiet:
        print("=" * 60)
        print("资产实测值")
        print("=" * 60)
        print("  决策点      : %d  （%s）" % (
            dp_total, " ".join("%s%d" % (k, dp_per[k]) for k in PREFIXES if k in dp_per)))
        print("  子条件      : %d" % sub_total)
        print("  陷阱条目    : %d  （T%d–T%d）" % (
            len(trap_nums), trap_nums[0] if trap_nums else 0, trap_nums[-1] if trap_nums else 0))
        print("  代码模板    : %d" % len(tpl_files))
        print()

    # ---------- 1. 内部自洽：陷阱编号必须连续 ----------
    if trap_nums:
        gaps = [n for n in range(trap_nums[0], trap_nums[-1] + 1) if n not in trap_nums]
        check(not gaps, "陷阱编号 T%d–T%d 连续无缺口%s" % (
            trap_nums[0], trap_nums[-1], "" if not gaps else "（缺 %s）" % gaps))
        check(trap_nums[0] == 1, "陷阱编号从 T1 起")
    else:
        check(False, "陷阱库未解析到任何 T 条目")

    # ---------- 2. 声明值 vs 实测值（SKILL 头部） ----------
    sk = parse_skill_header_counts(skill_txt)
    expect = {"决策点": dp_total, "子条件": sub_total, "陷阱": len(trap_nums), "模板": len(tpl_files)}
    for key, actual in expect.items():
        if key in sk:
            check(sk[key] == actual,
                  "SKILL 头部『资产清点』%s 声明 %d = 实测 %d" % (key, sk[key], actual))
        else:
            check(False, "SKILL 头部『资产清点』缺少 %s 的声明" % key)

    # ---------- 3. 声明值 vs 实测值（README 表） ----------
    rd = parse_readme_counts(readme_txt)
    for key, actual in expect.items():
        if key in rd:
            check(rd[key] == actual, "README 表 %s 声明 %d = 实测 %d" % (key, rd[key], actual))
        else:
            check(False, "README 表缺少 %s 的声明" % key)

    # ---------- 4. 跨文件一致 ----------
    for key in expect:
        if key in sk and key in rd:
            check(sk[key] == rd[key],
                  "SKILL 与 README 的 %s 一致（%d）" % (key, sk[key]))

    # ---------- 5. 决策点前缀覆盖（防止新前缀加了却漏登记） ----------
    stray = sorted({p for p, _ in re.findall(r"(?m)^([A-Z]{1,2})(\d+)【", skill_txt)} - set(PREFIXES))
    check(not stray, "无未登记前缀%s" % ("" if not stray else "：%s" % stray))

    # ---------- 输出 ----------
    if not args.quiet:
        print("=" * 60)
        for tag, msg in REPORT:
            print("  %-4s %s" % (tag, msg))
        print("=" * 60)

    if FAILS:
        print("资产清点不符 %d 项：" % len(FAILS))
        for f in FAILS:
            print("  - %s" % f)
        return 1
    print("资产清点一致：决策点 %d · 子条件 %d · 陷阱 %d · 模板 %d"
          % (dp_total, sub_total, len(trap_nums), len(tpl_files)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
