# -*- coding: utf-8 -*-
"""
scenario_planner.py —— 情景规划闭环模板库（华为杯解题 Skill v0.9）
==================================================================================
【适用场景】双碳/能源/政策路径规划类问题："情景参数化 → 核算-目标对标判定 →
           灵敏度分析 → 情景对比表"的闭环主链路（2023D 双碳路径规划）。
【溯源题号】2023D（H02）。
【核心纪律】① 每个情景必须三要素齐备才可发表: 政策锚点（达峰年/碳中和年/
           阶段约束值）+ 增长率/技术参数 + 对标口径（总量控 or 强度控 or 双控）；
           ② 对标判定用「最优年份 + 累计缺口」两个量，不只看终点值；
           ③ 灵敏度用 OAT 单因子 ±x% 出 tornado 图；要全局方差分解时上
           Sobol 灵敏度（见 uncertainty_optimization 模板）。
"""
import numpy as np

# ============================================================================
# 一、情景参数化（政策锚点 / 增长率）
# ============================================================================

def parametrize_scenario(base_year, base_value, anchors, horizon=2060, mode="log_linear"):
    """情景参数化: 政策锚点 + 段间规则 → 逐年排放路径。

    数学核心（锚点插值）: 已知点集 {基年值} ∪ {锚点(年, 值)}，在全区间上插值:
      mode="log_linear": 对数线性插值（段间按等削减率/增长率，IPCC 反推惯用）
                         —— 路径 = exp(interp(年, 锚点年, log(锚点值)))
      mode="linear":   线性插值（渐变政策）
    锚点前段由插值自动外推（等价按首个锚点斜率回推）；末锚点之后水平延展
    （np.interp 默认行为，即「目标达成后维持」的保守设定，论文需说明）。
    参数: base_year/base_value 基年排放；anchors [(年, 目标值), …] 政策锚点
          （升序、不重复）；horizon 末年；mode 插值模式。
    返回: dict{years, values} —— years 为 [基年..末年] 数组。
    适用场景: 高/中/低三套方案 = 同一函数三组锚点的三次调用，口径完全一致。
    溯源: 2023D（H02）。
    """
    years = np.arange(base_year, horizon + 1)
    pts_y, pts_v = [int(base_year)], [float(base_value)]
    for yy, vv in anchors:
        yy = int(yy)
        if base_year <= yy <= horizon and yy not in pts_y:
            pts_y.append(yy)
            pts_v.append(float(vv))
    order = np.argsort(pts_y)
    pts_y = np.asarray(pts_y)[order]
    pts_v = np.asarray(pts_v)[order]
    if mode == "log_linear":
        vals = np.exp(np.interp(years, pts_y, np.log(np.maximum(pts_v, 1e-12))))
    else:
        vals = np.interp(years, pts_y, pts_v)
    return {"years": years, "values": vals}


# ============================================================================
# 二、核算-目标对标判定（达标年 / 峰值 / 累计缺口 / 双控）
# ============================================================================

def compare_to_target(trajectory, target_path, years=None):
    """核算-目标对标: 首达年 / 峰值 / 累计缺口 三件套 + 逐年限差序列。

    数学核心:
      首达年   = 首个 traj ≤ target 的年份（None = 期内不达标）
      峰值     = (argmax 年份, max 值) —— 双碳语境即"达峰年/达峰值"
      累计缺口 = Σ_y max(traj_y − target_y, 0)（Δyear=1 时直接求和 ≈ 梯形积分）
    参数: trajectory/target_path 等长数组（强度或总量口径由调用方统一换算后
          传入同一数组）；years 年份标签（None 则用 0..n−1）。
    返回: dict{first_hit_year, peak_year, peak_value, cumulative_gap, gap_series}。
    溯源: 2023D（H02）—— 目标可达性判定环节。
    """
    t = np.asarray(trajectory, dtype=float)
    g = np.asarray(target_path, dtype=float)
    yrs = np.arange(t.size) if years is None else np.asarray(years)
    hit = np.flatnonzero(t <= g)
    peak_i = int(np.argmax(t))
    return {"first_hit_year": int(yrs[hit[0]]) if hit.size else None,
            "peak_year": int(yrs[peak_i]), "peak_value": float(t[peak_i]),
            "cumulative_gap": float(np.maximum(t - g, 0.0).sum()),
            "gap_series": np.maximum(t - g, 0.0)}


def dual_control_check(total_traj, intensity_traj, total_cap, intensity_cap, years=None):
    """双控达标判定: 总量与强度两条约束同时满足才算达标（2023D 口径）。

    数学核心: ok_y = (total ≤ cap_total) ∧ (intensity ≤ cap_intensity)，
    取首个同时成立的年份；返回全程违规年份清单便于画图。
    参数: 两条路径数组 + 两个上限（标量或逐年期数组）。
    返回: dict{total_ok, intensity_ok, both_ok_first_year, both_ok_years}。
    溯源: 2023D（H02）—— "总量与强度双控"是路径题的对标裁判。
    """
    T = np.asarray(total_traj, dtype=float)
    I = np.asarray(intensity_traj, dtype=float)
    yrs = np.arange(T.size) if years is None else np.asarray(years)
    ok = (T <= total_cap) & (I <= intensity_cap)
    ok_idx = np.flatnonzero(ok)
    return {"total_ok": bool(np.all(T <= total_cap)),
            "intensity_ok": bool(np.all(I <= intensity_cap)),
            "both_ok_first_year": int(yrs[ok_idx[0]]) if ok_idx.size else None,
            "both_ok_years": yrs[ok].tolist()}


# ============================================================================
# 三、灵敏度分析（OAT 单因子 ±x%）
# ============================================================================

def sensitivity_analysis(run_fn, factors, pct=0.1):
    """OAT 单因子灵敏度: 每个因子独立 ±x%，输出 tornado 数据表。

    数学核心: 对每个因子 f: y_lo = run(f×(1−pct))，y_hi = run(f×(1+pct))，
    range = |y_hi − y_lo|，span_pct = range / |y_nominal| —— 因子影响力排序。
    参数: run_fn(因子字典) → 标量指标（如 2030 排放、累计排放、缺口量）；
          factors (dict{因子名: 基准值})；pct (±百分数)。
    返回: dict{nominal, tornado: {因子名: {lo, hi, range, span_pct}}}；
    画 tornado: 按 range 降序 barh（docstring 附录 3 行示意）。
    溯源: 2023D（H02）—— 关键因素辨识环节，评审必看排序图。
    """
    nominal = float(run_fn(dict(factors)))
    tornado = {}
    for name, val in factors.items():
        lo_f, hi_f = dict(factors), dict(factors)
        lo_f[name] = val * (1.0 - pct)
        hi_f[name] = val * (1.0 + pct)
        y_lo = float(run_fn(lo_f))
        y_hi = float(run_fn(hi_f))
        tornado[name] = {"lo": y_lo, "hi": y_hi, "range": abs(y_hi - y_lo),
                         "span_pct": abs(y_hi - y_lo) / max(abs(nominal), 1e-12)}
    # 画 tornado 图: 按 range 降序后 matplotlib barh；或借用 pandas plot(kind='barh')
    return {"nominal": nominal, "tornado": tornado}


# ============================================================================
# 四、情景对比表输出
# ============================================================================

def scenario_comparison_table(scenarios, years=None):
    """情景对比表: 多情景路径 → (年份×情景)矩阵 + 逐情景关键指标行。

    数学核心: 所有情景先统一插值到共同年份轴（np.interp），横向拼接成矩阵；
    关键指标: 峰值年/峰值/累计排放（Σ，Δyear=1 近似积分）/末值/均值。
    参数: scenarios (dict{情景名: 该情景的 dict{years, values} 或路径数组})。
    返回: dict{matrix (n_years×n_情景), years, scenario_names, metrics}；
    论文出表: 行 = years，列 = 情景名，末尾附指标行（导出 Excel 用 pandas。
          列名即 scenarios.keys()）。
    溯源: 2023D（H02）—— 情景方案对比章节的标准交付物。
    """
    names = list(scenarios.keys())
    first = scenarios[names[0]]
    yrs = np.asarray(years) if years is not None else np.asarray(first["years"])
    mats = []
    for nm in names:
        s = scenarios[nm]
        v = s["values"] if isinstance(s, dict) else np.asarray(s)
        mats.append(np.interp(yrs, np.asarray(s["years"]), v))
    M = np.column_stack(mats)
    metrics = {}
    for j, nm in enumerate(names):
        v = M[:, j]
        metrics[nm] = {"peak_year": int(yrs[int(np.argmax(v))]),
                       "peak_value": float(v.max()),
                       "cum_total": float(v.sum()), "end_value": float(v[-1]),
                       "mean": float(v.mean())}
    return {"matrix": M, "years": yrs, "scenario_names": names, "metrics": metrics}