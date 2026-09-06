# -*- coding: utf-8 -*-
"""
clinical_longitudinal.py —— 临床纵向建模模板库（华为杯解题 Skill v0.9）
==================================================================================
【适用场景】患者个体差异 + 重复测量 + 有序结局的临床类问题：混合效应模型
           （随机截距/斜率捕捉患者异质性）→ 患者级分组 CV（防同一患者跨折
           泄漏）→ 有序结局建模（累积链接/取整回归 + QWK 评估）→ 混杂控制
           （倾向评分/IPW）（2023E 临床诊疗建模）。
【溯源题号】2023E（H03）。
【核心纪律】① 同一患者的所有随访行必须同折（GroupKFold）——随机 KFold 会
           把个体内相关性（ICC>0）泄漏进验证集，高估泛化；② 有序结局按
           序数口径建模 + QWK 评估，别当多分类独热处理（丢顺序信息）；
           ③ IPW 之后必须做平衡性检验（加权 SMD<0.1 方可接受），否则
           效应估计不具因果解释力。
"""
import numpy as np

# ============================================================================
# 一、混合效应模型（LME: 随机截距/随机斜率）
# ============================================================================

def lme_fit(df, formula, group_col, re_formula=None, REML=True):
    """线性混合效应模型: y ~ 固定效应 + 患者级随机效应（截距/斜率）。

    数学核心（LME 分块结构，b_i 为第 i 个患者的随机效应）:
      y_i = X_i·β + Z_i·b_i + ε_i，  b_i ~ N(0, G)，ε_i ~ N(0, σ²I)
      随机截距: Z_i = 1 列（re_formula="1"）；随机斜率: Z_i = [1, t_i]
      （re_formula="~ t"，允许患者间斜率异质，适合随访轨迹差异大的数据）
      推断: 似然比检验「有/无随机斜率」两种设定；AIC/BIC 比较写进论文。
    依赖: statsmodels（非 numpy 默认库，局部 import）。缺失降级: 装不上时
      退化为「固定效应 + 患者哑变量」OLS（等价随机截距的 GLS 近似），
      论文中必须注明口径差异；df 用 pandas，同样按需安装。
    参数: df (DataFrame，一行一次随访)；formula (如 "y ~ x1 + time")；
          group_col (患者 ID 列名)；re_formula (如 "~ time"，None=仅随机截距)。
    返回: statsmodels MixedLM 拟合对象，直接取 .params / .random_effects /
          .aic / .bic 出表；statsmodels 缺失时 raise 友好错误。
    溯源: 2023E（H03）—— 纵向随访数据的个体异质性建模。
    """
    try:
        from statsmodels.regression.mixed_linear_model import MixedLM
    except ImportError:
        raise ImportError(
            "statsmodels 未安装：pip install statsmodels；降级路径: 固定效应 + "
            "患者哑变量 OLS（GLS 近似），口径差异写进论文。")
    model = MixedLM.from_formula(formula, re_formula=re_formula or "1",
                                 groups=df[group_col], data=df)
    return model.fit(reml=REML)


# ============================================================================
# 二、患者级分组 CV（GroupKFold 防泄漏）
# ============================================================================

def patient_group_cv(model_fn, X, y, patient_ids, n_splits=5, metric="rmse"):
    """患者级分组交叉验证: 同一患者所有随访不跨折。

    与随机 KFold 的本质差异: 重复测量数据个体内相关（ICC > 0），随机切分
    会把同一患者的行同时放进训练与验证 → 泄漏高估；GroupKFold 按患者 ID
    折叠，结果才是患者外推（新患者预测）的真实水平。
    参数: model_fn(Xtr, ytr, ids_tr) → 带 .predict 的模型；patient_ids 与
          X 行对齐；metric ∈ {rmse, mae, auc, qwk}（qwk 见 quadratic_weighted_kappa）。
    返回: dict{folds: 每折指标, mean, std} —— 论文报 均值±标准差。
    溯源: 2023E（H03）—— 评估协议环节，评审核查防泄漏的第一处。
    """
    from sklearn.model_selection import GroupKFold
    X = np.asarray(X)
    y = np.asarray(y)
    ids = np.asarray(patient_ids)
    scores = []
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups=ids):
        m = model_fn(X[tr], y[tr], ids[tr])
        scores.append(_score_clinical(y[te], m.predict(X[te]), metric))
    return {"folds": scores, "mean": float(np.mean(scores)),
            "std": float(np.std(scores)) if len(scores) > 1 else 0.0}


def _score_clinical(y_true, y_pred, metric="rmse"):
    """临床指标分派器（内部函数）: rmse/mae/auc/qwk 单折打分。"""
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if metric == "rmse":
        return float(np.sqrt(np.mean((yt - yp) ** 2)))
    if metric == "mae":
        return float(np.mean(np.abs(yt - yp)))
    if metric == "auc":
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(yt, yp))
    if metric == "qwk":
        return float(quadratic_weighted_kappa(yt, np.round(yp)))
    raise ValueError("metric 仅支持 rmse/mae/auc/qwk")


# ============================================================================
# 三、有序目标建模（累积链接 / 取整回归 + QWK 评估）
# ============================================================================

def cumulative_link_fit(X, y, k=None):
    """有序累积链接模型（ordered logit）: P(y ≤ j|x) = σ(θ_j − xᵀβ)。

    数学核心（累积概率参数化天然保证序单调）:
      链接: logit P(y ≤ j|x) = θ_j − xᵀβ（j = 1..k−1，θ 递增）
      阈值单调处理: θ_j = θ₁ + Σ_{i≤j} exp(δ_i)（δ 无约束 ⇒ 优化自由参数）
      似然: P(y = j|x) = σ(θ_j − xβ) − σ(θ_{j−1} − xβ)，最大化对数似然
            （scipy.optimize.minimize, BFGS）
      预测: ŷ = 首个使累积概率 ≥ 0.5 的最小 j。
    对照备选: pip install mord 的 OrdinalLogistic 一行出结果 —— 论文里
      「自实现 vs 统计包一致」双路线论证。
    参数: X (n×p 特征)；y (0..k−1 整数有序标签)。
    返回: dict{beta, thresholds, loglik, n_iter}。
    溯源: 2023E（H03）—— 有序预后分级环节。
    """
    from scipy.optimize import minimize
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=int).ravel()
    k = int(y.max() + 1) if k is None else int(k)
    if k < 2:
        raise ValueError("有序标签至少 2 类")
    n, p = X.shape

    def unpack(th_all):                                  # 阈值参数化还原
        beta = th_all[:p]
        th0 = th_all[p]
        if k == 2:
            return beta, np.array([th0])
        delta = np.exp(th_all[p + 1:])
        th = np.concatenate([[th0], th0 + np.cumsum(delta)])
        return beta, th

    def neg_ll(th_all):
        beta, th = unpack(th_all)
        xb = X @ beta
        # 累积概率矩阵 (n, k−1): P(y ≤ j|x) = σ(θ_j − xβ)
        P = np.array([1.0 / (1.0 + np.exp(-(t - xb))) for t in th]).T
        P = np.clip(P, 1e-12, 1 - 1e-12)
        Pc = np.column_stack([P, np.ones(n)])            # 补 P(y ≤ k−1) = 1
        Pl = np.column_stack([np.zeros(n), P])           # 补 P(y ≤ −1) = 0
        prob = np.clip(Pc[np.arange(n), y] - Pl[np.arange(n), y], 1e-12, None)
        return -float(np.log(prob).sum())

    x0 = np.zeros(p + 1 + max(k - 2, 0))
    res = minimize(neg_ll, x0, method="BFGS")
    beta, th = unpack(res.x)
    return {"beta": beta, "thresholds": th, "loglik": -res.fun, "n_iter": res.nit}


def ordinal_rounding_regression(X, y_ordinal, base_model_fn=None, k=None):
    """取整回归（回归到有序类）: 连续潜变量回归 + 切点映射回序数类。

    数学核心: ①潜分数 y* 取类别代表值（默认 0..k−1 下标，可用类内秩均值/
    有序 logit 潜分改进）；②连续回归 y* ~ X（默认 Ridge）；③预测映射
    ŷ = 最近代表值（等价按切点取整、裁剪到 [0, k−1]）。
    切点学习（可选改进）: 在验证折上网格搜切点使 QWK 最大（折内选、折外测）。
    与直接分类相比保留顺序结构，小样本有序题更稳 —— 与累积链接互为对照。
    返回: (预测整型标签数组, 拟合模型)——评估用 quadratic_weighted_kappa。
    溯源: 2023E（H03）。
    """
    from sklearn.linear_model import Ridge
    y = np.asarray(y_ordinal, dtype=int).ravel()
    k = int(y.max() + 1) if k is None else int(k)
    if base_model_fn is not None:
        model = base_model_fn(X, y.astype(float))
    else:
        model = Ridge().fit(np.asarray(X, dtype=float), y.astype(float))
    score = model.predict(np.asarray(X, dtype=float))
    pred = np.clip(np.round(np.clip(score, 0, k - 1)), 0, k - 1).astype(int)
    return pred, model


def quadratic_weighted_kappa(y_true, y_pred, k=None, weights="quadratic"):
    """二次加权 Kappa（QWK）: 有序分类的标准一致性指标。

    数学核心（Cohen 加权 κ 的二次权重版）:
      O[i,j] = 混淆矩阵（k 级有序标签）；E[i,j] = 行和×列和 / N（期望一致）
      W[i,j] = (i−j)² / (k−1)²（完全一致 0、最远相差 1；线性版用 |i−j|/(k−1)）
      κ = 1 − Σ W·O / Σ W·E
    交叉核验: sklearn.metrics.cohen_kappa_score(yt, yp, weights='quadratic')
    与自实现一致时写进论文（两种实现互证）。
    参数: 真值/预测（整数数组，标签 0..k−1；越界自动裁剪）。
    返回: 标量 κ（越接近 1 越一致；k=1 退化时返回 NaN 需注明）。
    溯源: 2023E（H03）—— 有序预后分级的官方评估口径。
    """
    yt = np.clip(np.asarray(y_true, dtype=int).ravel(), 0, 2 ** 31 - 1)
    yp = np.clip(np.asarray(y_pred, dtype=int).ravel(), 0, 2 ** 31 - 1)
    if yt.shape != yp.shape:
        raise ValueError("真值与预测长度须一致（逐患者逐随访对齐）")
    k = int(max(yt.max(), yp.max()) + 1) if k is None else int(k)
    k = max(k, 1)
    O = np.zeros((k, k))
    np.add.at(O, (np.clip(yt, 0, k - 1), np.clip(yp, 0, k - 1)), 1.0)
    N = O.sum()
    if N == 0:
        return float("nan")
    E = np.outer(O.sum(axis=1), O.sum(axis=0)) / N
    if weights == "quadratic":
        W = (np.arange(k)[:, None] - np.arange(k)[None, :]) ** 2 / max((k - 1) ** 2, 1.0)
    else:
        W = np.abs(np.arange(k)[:, None] - np.arange(k)[None, :]) / max(k - 1, 1)
    denom = float((W * E).sum())
    if denom == 0:
        return float("nan")
    return float(1.0 - (W * O).sum() / denom)


# ============================================================================
# 四、混杂控制（倾向评分 / IPW + 平衡性检验）
# ============================================================================

def propensity_ipw(X, treatment, outcome=None, stabilize=True):
    """倾向评分 IPW: e(x) = P(T=1|x)，权重 w = T/e + (1−T)/(1−e)。

    数学核心（潜在结果框架）:
      倾向分 e(x) = LogisticRegression(x) 的 P(T=1|x)
      IPW 权重: w_i = T_i/e_i + (1−T_i)/(1−e_i)
      稳定化权重: w_i = T_i·P(T=1)/e_i + (1−T_i)·P(T=0)/(1−e_i)（方差更小，推荐）
      ATE = (Σ wTy)/(Σ wT) − (Σ w(1−T)y)/(Σ w(1−T))
      平衡性: 加权前后各协变量 SMD（见 _smd），目标全部 < 0.1
    参数: X 协变量；treatment 0/1；outcome 连续或二元结局（None 则只出权重）。
    返回: dict{weights, e, ate（若给 outcome）, smd_before, smd_after}。
    溯源: 2023E（H03）—— 组间基线差异的混杂控制环节（疗效对比题必做）。
    """
    from sklearn.linear_model import LogisticRegression
    T = np.asarray(treatment, dtype=float).ravel()
    e = LogisticRegression(max_iter=1000).fit(np.asarray(X, dtype=float), T)
    e = np.clip(e.predict_proba(np.asarray(X, dtype=float))[:, 1], 1e-6, 1 - 1e-6)
    w = T / e + (1 - T) / (1 - e)
    if stabilize:
        pt = float(T.mean())
        w = T * pt / e + (1 - T) * (1 - pt) / (1 - e)
    out = {"weights": w, "e": e}
    if outcome is not None:
        y = np.asarray(outcome, dtype=float).ravel()
        ate = (w * T * y).sum() / max((w * T).sum(), 1e-12) \
            - (w * (1 - T) * y).sum() / max((w * (1 - T)).sum(), 1e-12)
        out["ate"] = float(ate)
        out["smd_before"] = _smd(X, T)
        out["smd_after"] = _smd(X, T, w)
    return out


def _smd(X, T, w=None):
    """标准化均值差 SMD = |x̄₁−x̄₀| / √((s₁²+s₀²)/2)；给 w 时为加权版。

    判读: SMD < 0.1 → 平衡可接受（IPW 有效）；0.1–0.2 需在论文中讨论；
    > 0.2 → 倾向模型漏变量，回头加交互/非线性项。逐协变量返回数组。
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(T, dtype=bool).ravel()
    x1, x0 = X[T], X[~T]
    out = []
    for j in range(X.shape[1]):
        a, b = x1[:, j], x0[:, j]
        if w is None:
            m1, s1 = a.mean(), a.std(ddof=1)
            m0, s0 = b.mean(), b.std(ddof=1)
            out.append(abs(m1 - m0) / max(np.sqrt((s1 ** 2 + s0 ** 2) / 2), 1e-12))
        else:
            w1, w0 = np.asarray(w)[T], np.asarray(w)[~T]
            m1, m0 = (w1 * a).sum() / w1.sum(), (w0 * b).sum() / w0.sum()
            v1 = (w1 * (a - m1) ** 2).sum() / w1.sum()          # 加权方差(总体式)
            v0 = (w0 * (b - m0) ** 2).sum() / w0.sum()
            out.append(abs(m1 - m0) / max(np.sqrt((v1 + v0) / 2), 1e-12))
    return np.asarray(out)