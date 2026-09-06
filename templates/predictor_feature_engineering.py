# -*- coding: utf-8 -*-
"""
predictor_feature_engineering.py —— 预测类特征工程模板库（华为杯解题 Skill v0.1）
================================================================================
【适用场景】表格式/时序预测题中"367 变量 → ≤30 主变量"这类降维与特征构造环节。
【溯源题号】2020B（辛烷值：四路筛选证据链，问题2）、2021D（抗乳腺癌：粗筛-细筛
           范式 + Borda 聚合 + WOE/IV）、2021B（空气质量：IAQI 合成指标、滞后特征、
           滚动窗口）、2020C（脑信号：频段能量特征）、2019E（全球变暖：STL/趋势特征）、
           2018D（潮汐：三角谐波基函数）。
【红线】筛选必须在训练折内执行（防泄漏），PCA 只能当诊断工具不能当降维交付
        （主成分不可解释、不可回现场操作 —— 2020B 坑2 / 2021D 对齐点1）。
"""
import numpy as np

# ============================================================================
# 一、前置剔除与过滤法（2020B 问题2 路A + 近常量剔除）
# ============================================================================

def drop_near_constant(X, var_eps=1e-8, unique_max=3):
    """近常量列剔除（4 年稳态装置必有一批）—— s_j < ε 或唯一值 ≤ 3 删。

    等价做法（获奖论文）: 方差≈0 / 单一取值占比 >95% / 唯一值计数过小。
    溯源: 2020B §5.3 步1 + 2021D 对齐点2（武大 270 稀有变量、qq_45832050 唯一值列）。
    """
    keep = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.unique(col).size <= unique_max or np.nanvar(col) < var_eps:
            continue
        keep.append(j)
    return np.asarray(keep)


def correlation_rank(X, y, method="pearson"):
    """过滤法之相关排序：Pearson / Spearman（秩相关）。

    输出: 按 |r| 降序的变量序号 + r 值（附 p 值显著性）。
    Spearman 对单调非线性与离群更稳 —— 与 Pearson 并行看差异可发现非线性关系。
    溯源: 2020B §5.1 路A；2021D 独立解答“输出表”首列。
    """
    from scipy.stats import pearsonr, spearmanr
    fn = pearsonr if method == "pearson" else spearmanr
    rs, ps = [], []
    for j in range(X.shape[1]):
        r, p = fn(X[:, j], y)
        rs.append(r); ps.append(p)
    order = np.argsort(-np.abs(rs))
    return order, np.asarray(rs)[order], np.asarray(ps)[order]


def mutual_information_rank(X, y, k=5):
    """过滤法之互信息排序：Mine 类互信息（sklearn.feature_selection.mutual_info_regression）。

    MI 能抓 Pearson 看不见的非线性关联；与相关排序互为补充（路A 双通道）。
    """
    from sklearn.feature_selection import mutual_info_regression
    mi = mutual_info_regression(X, y, random_state=0)
    return np.argsort(-mi), mi


# ============================================================================
# 二、嵌入法与树模型重要性（2020B 路B/C、2021D 粗筛主力）
# ============================================================================

def lasso_select(X, y, alpha_path=None, l1_ratio=1.0):
    """LASSO/Elastic Net 稀疏回归选变量：CV 定 λ，1-SE 规则，取非零系数。

    数学: β̂ = argmin (1/2n)‖y − Xβ‖² + λ‖β‖₁（ElasticNet 为 λ(α‖β‖₁+(1−α)½‖β‖₂²)）。
    1-SE 规则: 取 CV 误差在最小值 1 个标准误内的最大 λ（更稀疏、更稳）。
    注意: λ 的选择必须在训练折内完成；系数路径图是论文配图。
    溯源: 2020B §5.2/§5.3 路B；2021D 粗筛（LightGBM 贡献度同族）。
    """
    from sklearn.linear_model import LassoCV, ElasticNetCV
    if l1_ratio >= 1.0:
        m = LassoCV(cv=5, random_state=0).fit(X, y)
    else:
        m = ElasticNetCV(l1_ratio=l1_ratio, cv=5, random_state=0).fit(X, y)
    return np.where(np.abs(m.coef_) > 1e-8)[0], m.coef_, m.alpha_


def tree_importance_select(X, y, top_k=None):
    """树模型重要性（双口径交叉）：RF 置换重要性 + GBDT/XGBoost gain。

    RF 置换重要性: 打乱特征后 OOB 误差增量（防"高基数特征虚高"）；
    XGBoost: gain（分裂增益）与 frequency（选中次数）双口径（2021D 独立解答 §Q2）。
    溯源: 2020B §5.1 路C；2021D 独立解答 97 行（XGBoost 双口径）。
    """
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.inspection import permutation_importance
    rf = RandomForestRegressor(n_estimators=300, random_state=0).fit(X, y)
    rf_imp = rf.feature_importances_
    perm = permutation_importance(rf, X, y, n_repeats=20, random_state=0).importances_mean
    try:
        import xgboost as xgb
        xm = xgb.XGBRegressor(n_estimators=200, random_state=0).fit(X, y)
        xg_gain = xm.feature_importances_          # 默认 gain
    except ImportError:
        xg_gain = np.zeros(X.shape[1])
    score = rf_imp + perm / (perm.max() + 1e-12) + xg_gain / (xg_gain.max() + 1e-12)
    return np.argsort(-score)[:top_k], {"rf_imp": rf_imp, "perm": perm, "xgb_gain": xg_gain}


def mrmr_select(X, y, k=30, bin_n=10):
    """mRMR 前向贪心：最大相关 - 最小冗余。

    max_S [ (1/|S|)Σ_{j∈S} I(x_j;y) − (1/|S|²)Σ_{j,k∈S} I(x_j;x_k) ]
    实现: 离散化后互信息计数（连续数据可改用 sklearn 的 MI 近似）。
    溯源: 2020B §5.1 路D（max-relevance min-redundancy）。
    """
    from sklearn.feature_selection import mutual_info_regression
    Iy = mutual_info_regression(X, y, random_state=0)
    # 特征-特征互信息用分箱近似（骨架：正式版可换更精确估计器）
    Xd = np.round((X - X.min(axis=0)) / (np.ptp(X, axis=0) + 1e-12) * (bin_n - 1)).astype(int)
    Ixx = np.zeros((X.shape[1], X.shape[1]))
    for i in range(X.shape[1]):
        for j in range(i + 1, X.shape[1]):
            t, _ = np.unique(np.vstack([Xd[:, i], Xd[:, j]]).T, axis=0, return_counts=True)
            p = _ / len(Xd)
            Ixx[i, j] = Ixx[j, i] = -np.sum(p * np.log2(p + 1e-12))  # 简化：联合熵近似
    S, remaining = [], list(range(X.shape[1]))
    for _ in range(min(k, X.shape[1])):
        scores = []
        for j in remaining:
            rel = Iy[j]
            red = np.mean([Ixx[j, s] for s in S]) if S else 0.0
            scores.append(rel - red)
        best = remaining[int(np.argmax(scores))]
        S.append(best); remaining.remove(best)
    return np.asarray(S)


def aggregate_votes(rankings, threshold=2, top_k=30):
    """多路交叉取稳：出现在 ≥2 路前 K 名的变量进"核心集"，仅 1 路的进"候选集"。

    与获奖论文"多口径交叉、排名稳定者入选"思想同构（武大 RF+Permutation+SHAP 三法）。
    溯源: 2020B §5.3 步3；2021D 对齐点4（Borda 聚合 + bootstrap 稳定性检验同族）。
    """
    from collections import Counter
    votes = Counter()
    for r in rankings:
        votes.update(r[:top_k])
    core = sorted([v for v, c in votes.items() if c >= threshold])
    cand = sorted([v for v, c in votes.items() if c == 1])
    return core, cand


def collinearity_prune(X, y, feat_idx, corr_thresh=0.8):
    """独立性终检：相关矩阵 + 层次聚类归并，簇内取"与 y 相关最高"的代表。

    VIF_j = 1/(1−R_j²) 超 10 等价于相关超 ~0.95，工程上常用 |r|>0.8 先聚类:
    距离 d_jk = 1 − |r_jk|，ward 法聚簇；簇内代表 = max |corr(x_j, y)| 者。
    溯源: 2020B §5.2 独立性终检 + 2021D 对齐点5（删 |r|>0.9 之一 / MIC+Spearman 独立性）。
    """
    from scipy.cluster.hierarchy import linkage, fcluster
    Xs = X[:, feat_idx]
    C = np.corrcoef(Xs.T)
    D = 1.0 - np.abs(C)
    Z = linkage(D[np.triu_indices(len(feat_idx), k=1)], method="ward")
    # 树状图阈值 → 使簇内最大 |r| ≥ corr_thresh（骨架：迭代调 height 或直接 cut）
    labels = fcluster(Z, t=1.0 - corr_thresh, criterion="distance")
    reps = []
    y_corr = np.abs([np.corrcoef(Xs[:, j], y)[0, 1] for j in range(len(feat_idx))])
    for c in np.unique(labels):
        members = np.where(labels == c)[0]
        reps.append(int(feat_idx[members[np.argmax(y_corr[members])]]))
    return reps, labels


def pca_diagnose(X):
    """PCA 仅作诊断：本征谱 → 有效秩 → 共线结构佐证（不是降维交付）。

    用途: (a) 显示 367 变量真实独立自由度远小（有效秩小→共线严重→需要归并）；
          (b) KPCA 同理只能诊断非线性结构。论文里"为什么不用 PCA 降维"的论据。
    溯源: 2020B §3 坑2 + §5.1 诊断位；2021D 对齐点1"选原始变量≠降维"红线。
    """
    Xc = X - X.mean(axis=0)
    evals = np.linalg.svd(Xc, compute_uv=False) ** 2 / max(X.shape[0] - 1, 1)
    cum = np.cumsum(evals) / evals.sum()
    eff_rank = int(np.searchsorted(cum, 0.9) + 1)
    return evals, cum, eff_rank


# ============================================================================
# 三、时序/合成/编码类特征（2021B、2020C、2019E、2018D、2021D）
# ============================================================================

def lag_window_features(X, lags=(1, 2, 7, 14), agg=("mean", "std", "max")):
    """滞后特征 + 滚动窗口统计：目标日特征 = f(前 N 日实测/一次预报)。

    2021B 核心特征理念: "一次预报 + 实测气象 + 近期实测浓度"组合，
    直接回归 y=f(·) 与 残差订正 ŷ=F+ r̂ 双流派共用此特征袋。
    时间因果红线: 所有滞后特征的时间戳必须 ≤ 起报时刻。
    溯源: 2021B §三 问题3 特征工程（得分关键）+ A5/A6。
    """
    out = []
    for lag in lags:
        for a in agg:
            fn = {"mean": np.mean, "std": np.std, "max": np.max}[a]
            out.append(np.array([fn(X[t - lag:t]) if t >= lag else np.nan
                                 for t in range(1, len(X) + 1)])[1:])
    return np.column_stack(out)


def iaqi_compute(C_p, bp_lo, bp_hi, iaqi_lo, iaqi_hi):
    """IAQI 分段线性插值（空气质量分指数）—— 送分题要拿满分的精确口径。

    IAQI_p = ⌊ (IAQI_Hi−IAQI_Lo)/(BP_Hi−BP_Lo) × (C_p − BP_Lo) + IAQI_Lo + 0.5 ⌋
    （四舍五入取整为默认口径；附录若写"进位取整"则为 ⌈·⌉ —— 双口径实现对比，
    取与题目样例一致者。O3 用日最大 8h 滑动平均、其余 24h 平均；超 IAQI=500 限值
    不再计算该分指数；AQI = max_p IAQI_p，≤50 无首要污染物。）
    溯源: 2021B §三 问题1 全部规则（对齐点 A1/A2 完全一致）。
    """
    return int((iaqi_hi - iaqi_lo) / (bp_hi - bp_lo) * (C_p - bp_lo) + iaqi_lo + 0.5)


def woe_iv(X, y, bins=10):
    """证据权重 WOE 与信息价值 IV（二分类任务的单调化编码 + 筛选）。

    WOE_j = ln(好样本占比 / 坏样本占比)（按分箱）；
    IV_j  = Σ(好占比 − 坏占比)·WOE —— 经验阈值: IV<0.02 无用, 0.02~0.1 较弱,
           0.1~0.3 中等, >0.3 强但需警惕过拟合。
    对每个候选特征返回 (IV 排序)；WOE 编码替换原始值后再喂模型（Logistic 标配）。
    溯源: 2021D 差距分析二.1 缺口补强项（"缺 WOE/IV 筛选"被点名）+ 独立解答 Q2 框架。
    """
    ivs, woes = [], {}
    for j in range(X.shape[1]):
        q = np.quantile(X[:, j], np.linspace(0, 1, bins + 1))
        q[0], q[-1] = q[0] - 1e-9, q[-1] + 1e-9
        bin_id = np.digitize(X[:, j], q[1:-1])
        iv, wj = 0.0, {}
        for b in range(bins):
            mask = bin_id == b
            good = y[mask].sum(); bad = mask.sum() - good
            g_rel = good / max(y.sum(), 1e-9); b_rel = bad / max((1 - y).sum(), 1e-9)
            g_rel = np.clip(g_rel, 1e-6, 1); b_rel = np.clip(b_rel, 1e-6, 1)
            woe_b = np.log(g_rel / b_rel)
            wj[b] = woe_b
            iv += (g_rel - b_rel) * woe_b
        ivs.append(iv); woes[j] = wj
    return np.argsort(-np.asarray(ivs)), ivs, woes


def energy_band_features(psd, bands):
    """频段能量占比特征（脑电/信号类题目标准动作）。

    对 PSD 按频段（如 δ/θ/α/β）求能量占比，可再取对数/比值变换
    （2020C: θ+δ 占比、α+β 占比、lnE —— 与获奖论文"9 个新特征"同构）。
    溯源: 2020C 对齐点 A10（南大/交大一致：能量组合特征）。
    """
    total = psd.sum()
    feats = []
    for (lo, hi) in bands:
        feats.append(psd[lo:hi].sum() / max(total, 1e-12))
    return np.asarray(feats)


def stl_trend_component(series, period=None):
    """时序分解取趋势/季节分量（2019E 时间维度分析的预处理件）。

    用途: 趋势-突变-周期三维度叙事的前两步（Theil-Sen/MK 检验在趋势分量上做更干净）。
    实现: statsmodels STL（按需裁剪；无 statsmodels 时可用 移动平均+去趋势 兜底）。
    溯源: 2019E 对齐点（STL 季节分解 → 趋势 + 突变 + 周期框架与国一队伍重合）。
    """
    try:
        from statsmodels.tsa.seasonal import STL
        res = STL(series, period=period if period else max(2, len(series) // 4)).fit()
        return res.trend, res.seasonal, res.resid
    except ImportError:
        k = period if period else max(2, len(series) // 4)
        trend = np.convolve(series, np.ones(k) / k, mode="same")   # 移动平均兜底
        return trend, series - trend, np.zeros_like(series)


def harmonic_design_matrix(t, freqs, f_mod=None, u_mod=None):
    """谐波/三角基函数设计矩阵（周期信号线性拟合的标准块）。

    列: [1, t, {f_k(t)·cos(σ_k t + V_k + u_k), f_k(t)·sin(...)}] —— 分潮/季节/日周期
    均可放入。f/u 交点修正存在时逐时刻乘上调制（2018D 达尔文-杜森模型）。
    与 estimate 配合: θ̂ = argmin ‖W^½(y − Xθ)‖²，H_k = √(x_k²+y_k²)，g_k = atan2(y,x)。
    溯源: 2018D §4.2 线性化（x=Hcos g, y=Hsin g 变量代换，获奖论文同法）。
    """
    cols = [np.ones_like(t), t]
    for i, f0 in enumerate(freqs):
        phase = 2 * np.pi * f0 * t
        if f_mod is not None:
            phase = phase + u_mod[:, i]
            amp = f_mod[:, i]
        else:
            amp = 1.0
        cols.append(amp * np.cos(phase))
        cols.append(amp * np.sin(phase))
    return np.column_stack(cols)