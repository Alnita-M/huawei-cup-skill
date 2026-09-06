# -*- coding: utf-8 -*-
"""
predictor_data_audit.py —— 预测类数据审计模板库（华为杯解题 Skill v0.1）
=====================================================================
【适用场景】表格式数据预测题（含 325×367 类大表、时间序列、多源数据）的
           "动手建模前的数据体检" 环节：缺失/异常/口径/对齐/分布五板斧。
【溯源题号】2020B（汽油辛烷值：预处理五件套，问题1）、2021D（抗乳腺癌：KDE
           train/test 分布一致性，问题2 前置）、2018D（潮汐：采样混叠与
           Rayleigh 分辨、UTC 相位约定，问题1）、2021B（空气质量：小时数据
           有效性规则、时间因果检查）、2019D（行驶工况：物理判废 + 时间断点）。
【使用方法】函数骨架，按 docstring 约定补全业务；pandas 为主力、numpy/scipy 兜底。
"""
import numpy as np

# ============================================================================
# 一、缺失值审计（2020B 问题1 —— 五件套的第 1-3 步）
# ============================================================================

def missing_profile(X, col_names=None):
    """缺失率全景：按列（位点）与按行（时间节点/样本）双向统计。

    列向: r_j = miss_j / n_rows —— 全空(r=1)删列；r_j > θ_miss 删列（残缺较多）；
          否则保留进入补全。θ_miss 取值须引用题目/附件方法原文。
    行向: 同一时刻几乎全部列缺失（装置停工/仪表集体离线）→ 整行剔除，不参与均值。
    返回: (col_miss_ratio, row_miss_ratio) 两个 np.ndarray。
    溯源: 2020B §4.2 步骤1-2。
    """
    X = np.asarray(X, dtype=float)
    col_ratio = np.isnan(X).mean(axis=0)
    row_ratio = np.isnan(X).mean(axis=1)
    return col_ratio, row_ratio


def impute_simple(X, method="mean", neighbor_span=3):
    """缺失补全骨架：列均值 / 列中位数 / 相邻行线性插值。

    决策点（论文要写理由）: 附件方法指定均值补全则忠实执行；
    若缺失成连续段且工艺量缓变，相邻线性插值更合理 —— 做敏感性对照，不擅自改主口径。
    参数: X (n, p) 含 NaN。
    溯源: 2020B §4.2 步骤3（+2021B 短时缺失插补 A9、2019D 短补长弃同族）。
    """
    X = X.copy()
    for j in range(X.shape[1]):
        col = X[:, j]
        nan_idx = np.isnan(col)
        if nan_idx.all():
            continue                          # 整列空应在 missing_profile 已删
        if method in ("mean", "median"):
            fill = np.nanmean(col) if method == "mean" else np.nanmedian(col)
            col[nan_idx] = fill
        elif method == "linear":
            ok = ~nan_idx
            col[nan_idx] = np.interp(np.flatnonzero(nan_idx),
                                     np.flatnonzero(ok), col[ok])
        X[:, j] = col
    return X


# ============================================================================
# 二、异常值审计（2020B 问题1 —— 五件套的 4-5 步；2019D 物理判废）
# ============================================================================

def clip_bounds(X, lo, hi, mode="clip"):
    """最大最小限幅（硬可行域截断）。

    mode='clip': x' = clip(x, L, U) —— 幅度小的仪表毛刺直接截断；
    mode='flag': 返回越界掩码 —— 持续越界（整段超物理极限）判坏值剔除而非截断，
                 避免把系统性错误留在均值里。判别 = 越界时长/幅度。
    参数: lo, hi (p,) 或标量，来自题目附件给出的操作范围（如 2020B 附件四）。
    溯源: 2020B §4.2 步骤4（附件四操作范围）+ A7。
    """
    X = np.asarray(X, dtype=float)
    out_of_range = (X < lo) | (X > hi)
    if mode == "flag":
        return out_of_range
    return np.clip(X, lo, hi)


def iterative_3sigma_outlier(X, robust=False, max_rounds=5):
    """迭代 3σ（拉依达准则）异常剔除 —— 必须迭代，单遍有遮蔽效应。

    步骤: 1) 正态性粗检（Q-Q 图/SW 检验）—— 拉依达以正态为前提；
         2) 严重偏态 → 稳健版: 中位数 ± 3·1.4826·MAD（替代均值±3σ）；
         3) 剔除 |x−μ| > 3σ 后重算 μ,σ 再剔，迭代至无新剔除；
         4) 剔除点数占比 >5% → 整列质量差，考虑删除（记录在案）。
    先限幅后 3σ 的顺序更稳（极限值不先截断会拉大 σ 拖累剔除灵敏性）——报告写明。
    溯源: 2020B §4.2 步骤5 + A1。
    """
    from scipy.stats import median_abs_deviation
    X = np.asarray(X, dtype=float)
    ok = np.ones(X.shape[0], dtype=bool)
    for _ in range(max_rounds):
        col = X[ok]
        if robust:
            mu = np.median(col)
            sd = 1.4826 * median_abs_deviation(col)
        else:
            mu, sd = col.mean(), col.std(ddof=1)      # 贝塞尔公式 s
        bad = np.abs(X - mu) > 3 * sd
        if not bad.any():
            break
        ok &= ~bad
    return ok                                          # True = 保留


def physical_range_sanity(v, v_max=120.0, a_upper=3.97, a_lower=-8.0):
    """物理规律判废（速度/加速度类时序）—— 领域规则兜底。

    超速 v>120 → 判废；加速度上界 (100/3.6)/7≈3.97、下界 −8 → 判废。
    处置原则（2019D 对齐点 A6/A7）: 孤立异常优先邻域中位数修复，
    连片异常才删除；修复优于删除（保数据连续性）。
    溯源: 2019D 对齐点 A3/A6/A7（与获奖论文阈值一致）。
    """
    a = np.gradient(v)
    bad = (v > v_max) | (a > a_upper) | (a < a_lower)
    return bad


# ============================================================================
# 三、口径与对齐审计（2018D / 2021B / 2020B —— 最容易被忽略的"系统性错"源）
# ============================================================================

def audit_time_alignment(t_obs, t_ref, freq_expected=None):
    """时间对齐三查：时区统一 / 采样周期 / 因果性（特征时间戳 ≤ 起报时刻）。

    ① 时区/相位基准: 两地数据若一份 UTC 一份地方时（如 2018D 验潮站迟角
       GMT+8 陷阱 → 系统性 120° 相位偏差），先统一基准再对齐；
    ② 采样周期: 检查实际间隔 vs 期望（如 T/P 重复周期 9.9156 d）；
       核对成样窗口口径（2020B: RON 测量前 2h 均值 = 40 节点 × 3min）；
    ③ 时间因果: 预测任务中"特征时间戳 ≤ 预测起报时刻"逐列核对防未来信息泄漏
       （2021B A6 隐性评分门槛：只用 7 点前实测与当日及以前运行的一次预报）。
    返回: dict{时区标记, 采样间隔中位数/众数, 因果违规列清单}。
    溯源: 2018D A8 假设与 D7 偏离点；2021B A6；2020B 坑3。
    """
    gaps = np.diff(np.sort(t_obs))
    median_gap = np.median(gaps)
    return {
        "median_gap": median_gap,
        "freq_expected": freq_expected,
        "gap_matches": (freq_expected is None) or np.isclose(median_gap, freq_expected, rtol=0.05),
        "note": "统一时区基准 + 特征时间戳≤起报时刻 因果检查表",
    }


def aliasing_period(freqs, sample_period):
    """采样混叠周期表（欠采样信号诊断）—— 回答"资料长度够不够"。

    混叠表观频率: f_a = |f − k/T|，k = round(f·T)（cos 偶对称，取折叠到 [0, Nyquist] 后之差）
    Rayleigh 分辨判据: 分辨两分潮所需最短资料长度 T_min ≈ 1/|f_a1 − f_a2|。
    例（T=9.9156 d）: M2 混叠到 62.1 d、S2→58.7 d、M2−S2 需 ≈3 年、K1−P1 需 ≈0.5 年。
    结论用于: 数据长度是否足以分辨目标信号、能否忽略交点修正（18.61 a 周期）。
    溯源: 2018D §3 全节（"能否有效提取某分潮取决于资料长度"的正面回答）。
    """
    fa = np.abs(freqs - np.round(freqs * sample_period) / sample_period)
    return np.array([1.0 / f_ if f_ > 0 else np.inf for f_ in fa])   # 表观周期


def rayleigh_resolution(freq_a, freq_b):
    """两信号表观频率差 → 所需最短记录长度 T_min ≈ 1/|f_a − f_b|。"""
    return 1.0 / max(abs(freq_a - freq_b), 1e-12)


def validity_rules_24h(hourly_values, min_valid_hours=20, o3_8h_min=6):
    """日均值有效性规则（空气质量题 Q1 的"数据不足"判定）。

    24h 平均要求当日有效小时数 ≥ 20；O3 的每个 8h 滑动窗口要求 ≥6 有效小时，
    不足则窗口作废；整日无法产生有效窗口 → 标"缺失"而非强行给数。
    溯源: 2021B §二 假设 A7 + §三 问题1 数据有效性规则【不确定2】。
    """
    valid = ~np.isnan(hourly_values)
    return {"day_valid": valid.sum() >= min_valid_hours,
            "o3_windows": [valid[i:i + 8].sum() >= o3_8h_min
                           for i in range(len(valid) - 7)]}


# ============================================================================
# 四、数据分布审计（2021D 偏离点1 —— 训练/测试分布漂移；2018D 前置谱诊断）
# ============================================================================

def kde_distribution_check(X_train, X_test, feat_names=None, ks_alpha=0.05):
    """训练/测试特征分布一致性：逐列 KDE 对比 + KS 检验，剔除漂移显著者。

    动机（获奖论文实证）: test 集 50 个化合物若与训练分布偏移，模型精度会被
    系统性高估（2021D 差距分析"低成本、高价值实操细节"）。
    做法: 对每个候选特征画 train/test KDE（或做 KS 检验），剔除漂移显著者后建模。
    返回: dict{evicted: 漂移显著列序号, ks_p: 每列 p 值}。
    溯源: 2021D 差距分析 二.1（qq_45832050 做法）+ 独立解答 §2 假设 A5。
    """
    from scipy.stats import ks_2samp
    pvals = []
    for j in range(X_train.shape[1]):
        pvals.append(ks_2samp(X_train[:, j], X_test[:, j]).pvalue)
    evicted = [j for j, p in enumerate(pvals) if p < ks_alpha]
    return {"evicted": evicted, "ks_p": np.array(pvals),
            "note": "evicted 列在分布层面漂移，建议剔除或做分域建模"}


def lomb_scargle_diagnosis(t, y, fmin=0.001, fmax=1.0):
    """非均匀采样信号的谱诊断（潮汐/天文采样），验证目标频率是否存在。

    若谱峰落在混叠表观周期处 → 目标信号确实存在（2018D 假设 A1 验证）；
    若谱峰极弱 → 检查数据说明（可能为模型残差而非原始信号）。
    实现: 直接调 scipy.signal.lombscargle（不配平、不编造）。
    溯源: 2018D §4.3 前置诊断。
    """
    from scipy.signal import lombscargle
    freqs = np.linspace(fmin, fmax, 10000)
    pgram = lombscargle(t, y, freqs)
    return freqs, pgram


# ============================================================================
# 五、审计报告汇总
# ============================================================================

def audit_report(X, t=None, lo=None, hi=None, X_test=None):
    """一键审计总览：缺失 → 异常 → 限幅 → 分布 → 时间，输出结构化报告。

    典型调用（模拟 2020B 问题1 流水线）:
      X 清洗 = clip(迭代3σ(补全(删列(删行(X)))))，最后列均值成样（2h 窗口）。
    溯源: 2020B §4.2 全管道 + 验证方法 1-4（结构一致性/口径一致性/敏感性/顺序反演）。
    """
    col_r, row_r = missing_profile(X)
    rep = {"列缺失率": col_r, "行缺失率": row_r,
           "删除建议": {"全空列": np.where(col_r >= 1.0)[0].tolist(),
                        "行全缺": np.where(row_r >= 0.95)[0].tolist()}}
    if lo is not None and hi is not None:
        rep["越界占比"] = float(np.mean((X < lo) | (X > hi)))
    if X_test is not None:
        rep["分布漂移"] = kde_distribution_check(X, X_test)
    if t is not None:
        rep["时间"] = audit_time_alignment(t)
    return rep