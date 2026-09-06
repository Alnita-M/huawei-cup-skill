# -*- coding: utf-8 -*-
"""
emissions_ledger.py —— 双碳核算矩阵模板库（华为杯解题 Skill v0.9）
==================================================================================
【适用场景】区域/行业碳排放核算类问题："活动数据 × 排放因子 × GWP → 部门/品种
           核算矩阵 → 汇总 → 勾稽校验 → 数据缺口标注"主链路（2023D 双碳核算）。
【溯源题号】2023D（H02）。
【核心纪律】① 处处可勾稽: 任意分项和在数值上必须等于总量（表内自洽 + 表间
           互洽），todo 差值进论文"差距表"；② CO₂e 口径先锁定 GWP 版本
           （IPCC AR4/AR5 甲烷系数差异可达数倍，先写明再用）；
           ③ 缺口不掩盖: 缺失/异常/估计值显式标注处理口径，禁止静默填零。
"""
import numpy as np

# ============================================================================
# 一、核算矩阵构建（部门 × 品种 × 因子）
# ============================================================================

def build_emission_ledger(activity, ef_table, gwp=None, names=None):
    """部门×品种×因子核算矩阵: E[s,g] = Σ_f A[s,f]·EF[f,g]（CO₂e = E·GWP）。

    数学核心（核算恒等式，三次矩阵乘完成全部核算）:
      活动水平 A: S×F  —— S 部门 × F 能源品种/活动类别（实物量）
      排放因子 EF: F×G —— 品种 × 温室气体（每单位实物量的气体排放）
      排放矩阵  E: S×G —— E[s,g] = Σ_f A[s,f]·EF[f,g] = A @ EF
      CO₂e 汇总:  E @ GWP（GWP 为 G 维列向量；CO₂ 本身 GWP=1，其他气体
                  按其百年/二十年潜势折算，版本口径写进论文）
    参数: activity (S×F 数组；若源数据是长表，先透视成 S×F)；
          ef_table (F×G 数组)；gwp (G 维数组，None= 只算实物气体不折 CO₂e)；
          names (可选 dict{sector, fuel, gas}: 行/列标签列表，便于出表)。
    返回: dict{E_by_gas (S×G), co2e_by_sector (S 维或 None), total (标量),
          sector_names, fuel_names, gas_names}。
    适用场景: 拿到活动数据与因子表后的第一步 —— 后续情景/对比/勾稽全部
              以此为单一数据源，杜绝各表口径不一致。
    溯源: 2023D（H02）。
    """
    A = np.asarray(activity, dtype=float)
    EF = np.asarray(ef_table, dtype=float)
    if A.shape[1] != EF.shape[0]:
        raise ValueError("活动数据列数必须等于排放因子表行数（品种维对齐）")
    E = A @ EF                                          # (S, G)
    co2e = (E @ np.asarray(gwp, dtype=float)) if gwp is not None else None
    names = names or {}
    return {"E_by_gas": E,
            "co2e_by_sector": co2e,
            "total": float(co2e.sum()) if co2e is not None else float(E.sum()),
            "sector_names": names.get("sector", None),
            "fuel_names": names.get("fuel", None),
            "gas_names": names.get("gas", None)}


def aggregate_emissions(E, gwp=None):
    """任意维度汇总: 部门合计 / 气体(含 CO₂e)合计 / 总量 三件套 + 结构占比。

    数学核心: 部门 CO₂e = E@GWP；分气体 CO₂e = (Eᵀ·1) ⊙ GWP；
    总量 = 1ᵀ·E·GWP；占比 = 分项/总量（论文"XX 部门贡献 XX%"就此表出）。
    参数: E (S×G 排放矩阵)；gwp (G 维，None 则只汇总实物量)。
    返回: dict{by_sector, by_gas, total, share_sector, share_gas}。
    溯源: 2023D（H02）—— 结构与贡献份额环节。
    """
    E = np.asarray(E, dtype=float)
    if gwp is not None:
        g = np.asarray(gwp, dtype=float)
        by_sector = E @ g                                # (S,)
        by_gas = E.sum(axis=0) * g                       # (G,) 实物×GWP
        total = float(by_sector.sum())
    else:
        by_sector = E.sum(axis=1)
        by_gas = E.sum(axis=0)
        total = float(E.sum())
    return {"by_sector": by_sector, "by_gas": by_gas, "total": total,
            "share_sector": by_sector / max(total, 1e-12),
            "share_gas": by_gas / max(total, 1e-12)}


def carbon_indicators(co2e_total, gdp=None, population=None):
    """双控三指标: 总量 / 排放强度(CO₂e÷GDP) / 人均排放 —— 情景对标常用分母。

    注意: 单位口径由调用方统一（如 万吨 CO₂e、亿元 GDP），强度单位
    「吨 CO₂e/万元 GDP」或「万吨/亿元」要写进表头，评审最看单位一致性。
    返回: dict{"total", "intensity_per_gdp"（None 缺 GDP）, "per_capita"}。
    溯源: 2023D（H02）—— 双控口径下的总量与强度。
    """
    out = {"total": float(co2e_total)}
    if gdp is not None:
        out["intensity_per_gdp"] = float(co2e_total / gdp)
    if population is not None:
        out["per_capita"] = float(co2e_total / population)
    return out


# ============================================================================
# 二、勾稽校验（分项和 = 总量）
# ============================================================================

def cross_foot_check(parts, total, rtol=1e-6, atol=1e-9):
    """勾稽校验: 分项和 vs 总量必须一致（容忍带内算通过）。

    数学核心: |Σ parts − total| ≤ atol + rtol·|total|（numpy 宽容差语义）；
    输出差异明细 diff/rel_diff —— 论文「分项和=总量」的差距表直接由此生成。
    适用: 核算表自洽、拆分表回加、情景表对账、报表交叉验证（表间互洽）。
    返回: dict{sum_parts, total, diff, rel_diff, passed}。
    溯源: 2023D（H02）—— 核算质量环节，评审核查的第一处。
    """
    s = float(np.asarray(parts, dtype=float).sum())
    diff = s - float(total)
    tol = atol + rtol * abs(float(total))
    rel = diff / abs(float(total)) if abs(float(total)) > atol else None
    return {"sum_parts": s, "total": float(total), "diff": diff,
            "rel_diff": rel, "passed": bool(abs(diff) <= tol)}


def check_ledger_consistency(rows, expected_totals, row_labels=None):
    """批量化勾稽: 对一张「分项明细表」逐行/逐列与总量对账，返回违规清单。

    参数: rows (k×n 数组：k 个分项 × n 个对账列)；expected_totals
          (n 维或标量：各列期望总量)。
    返回: dict{violations: [(行标签或下标, 列, diff, rel_diff), …], n_ok, n_bad}。
    溯源: 2023D（H02）—— 大表对账场景的批量出口。
    """
    rows = np.asarray(rows, dtype=float)
    exp = np.full(rows.shape[1], float(expected_totals)) if np.ndim(expected_totals) == 0 \
        else np.asarray(expected_totals, dtype=float)
    violations = []
    for i in range(rows.shape[0]):
        chk = cross_foot_check(rows[i], exp[i])
        if not chk["passed"]:
            label = row_labels[i] if row_labels is not None else i
            violations.append((label, chk["diff"], chk["rel_diff"]))
    return {"violations": violations, "n_ok": rows.shape[0] - len(violations),
            "n_bad": len(violations)}


# ============================================================================
# 三、数据缺口标注（缺口透明化，禁止静默填零）
# ============================================================================

def mark_data_gaps(activity, valid_mask=None, extra_rules=None):
    """缺口标注: 定位缺失/异常/无效单元格并给处理口径，附覆盖率统计。

    数学核心: 逐格判定 → 标注矩阵（标签: 缺失(NaN)/无效(valid_mask 为假)/
    自定义规则），覆盖率 = 有效格占比 —— 论文「数据质量」小节的数据来源。
    参数: activity (S×F 数组)；valid_mask (同形状布尔，None= 全有效)；
          extra_rules (dict{标签: 谓词函数 A→bool 数组}，如 {"负值异常": lambda A: A < 0})。
    返回: dict{flag_matrix, flagged_cells: [(i, j, 标签), …], coverage, n_flagged}。
    溯源: 2023D（H02）—— 数据质量与缺口处理环节。
    """
    A = np.asarray(activity, dtype=float)
    flags = np.full(A.shape, "", dtype=object)
    mask = np.ones(A.shape, dtype=bool) if valid_mask is None \
        else np.asarray(valid_mask, dtype=bool)
    flags[np.isnan(A)] = "缺失(NaN)"
    flags[~mask & ~np.isnan(A)] = "无效(valid_mask)"
    if extra_rules:
        for label, pred in extra_rules.items():
            hit = np.asarray(pred(A), dtype=bool)
            flags[hit & (flags == "")] = str(label)
    idx = np.argwhere(flags != "")
    return {"flag_matrix": flags,
            "flagged_cells": [(int(i), int(j), flags[i, j]) for i, j in idx],
            "coverage": float(1.0 - len(idx) / max(A.size, 1)),
            "n_flagged": len(idx)}