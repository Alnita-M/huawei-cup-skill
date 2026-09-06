# -*- coding: utf-8 -*-
"""
ensemble_forecast.py —— 集合预报模板库（华为杯解题 Skill v1.0）
==================================================================================
【适用场景】预测/短临外推类题的不确定性量化："多成员生成（多初值/多模型/扰动）
           → 分位聚合（P10/P50/P90）→ 区间输出 → 平流一致性检查"主链路
           （2025D 低空湍流模型 e：半拉格朗日平流外推 + 集合扰动 ±10–20% 幅度、
           ±10° 方向 → P10/P50/P90）。
【溯源题号】2025D（F04）。配套决策点：SKILL.md STEP R R5（集合预报形态）、
           R4（时空场时间维平滑）、3.2 M5（多法并列融合优先）。
【核心纪律】① 不确定性量化输出形态优先"集合"（成员点列 + 分位区间），
           可视化友好、直接服务下游决策（航路稳健性/阈值判定）；
           ② 扰动幅度/方向要按题面物理量规格声明（如幅度 ±10–20%、方向 ±10°），
           与去相关尺度/预测区间法并列对照（U1 类方法），不默认唯一正确；
           ③ 集合区间必做可靠性检验（覆盖率的 PIT 直方图/经验覆盖率），
           防止区间系统性过窄或过宽。
"""
import numpy as np

# ============================================================================
# 一、成员生成（扰动 / 多初值 / 多模型）
# ============================================================================

def perturb_vector_members(base_u, base_v, n_members=20, amp_range=(0.10, 0.20),
                           dir_range_deg=10.0, rng=None):
    """幅度+方向摄动生成集合成员（标量场用 perturb_scalar_members）。

    数学核心（2025D 获奖形态）: 对控制预报 (u, v) 每个格点做乘性幅度扰动与
    旋转扰动:
        u_i = A_i·(u·cosθ_i − v·sinθ_i),  v_i = A_i·(u·sinθ_i + v·cosθ_i)
      A_i ~ U(1−amp_max, 1+amp_max) （默认 ±10–20%，amp_range 取上下限）
      θ_i ~ U(−dir_max, +dir_max)     （默认 ±10°）
    每个成员用一个 (A_i, θ_i) 全局摄动（位移场整体不确定性），保留空间相关
    结构；若要逐格点独立扰动，把 rng 换 per-cell 抽样即可（噪声更大，慎用）。
    参数: base_u/base_v 控制预报 u/v 分量（同形状 ndarray）；n_members 成员数；
          amp_range 幅度扰动上下限；dir_range_deg 方向扰动半幅（度）；
          rng np.random.Generator（None 则默认种子，保证可复现）。
    返回: dict{members_u: (n_members,H,W), members_v: (n_members,H,W),
          amps, thetas} —— amps/thetas 记录每成员扰动参数（论文可报告分布）。
    溯源: 2025D（F04）—— 集合扰动形态（±10–20% 幅度、±10° 方向）。
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    u = np.asarray(base_u, dtype=float)
    v = np.asarray(base_v, dtype=float)
    amps = rng.uniform(1 - amp_range[0], 1 + amp_range[1], size=n_members)
    thetas = np.deg2rad(rng.uniform(-dir_range_deg, dir_range_deg, size=n_members))
    mu = np.empty((n_members,) + u.shape)
    mv = np.empty((n_members,) + v.shape)
    for i in range(n_members):
        c, s = np.cos(thetas[i]), np.sin(thetas[i])
        mu[i] = amps[i] * (u * c - v * s)
        mv[i] = amps[i] * (u * s + v * c)
    return {"members_u": mu, "members_v": mv, "amps": amps, "thetas": thetas}


def perturb_scalar_members(base, n_members=20, amp_range=(0.10, 0.20),
                           additive_sigma=None, rng=None):
    """标量场集合成员：乘性幅度扰动（+可选加性噪声）。

    数学核心: member_i = base·A_i + η_i
      A_i ~ U(1−amp_min, 1+amp_max)；η_i ~ N(0, additive_sigma²)（可选，
      用于模拟仪器噪声，默认 None=纯乘性扰动）。
    参数: base 控制预报标量场；n_members 成员数；amp_range 乘性扰动上下限；
          additive_sigma 加性噪声标准差（None=不加）。
    返回: dict{members: (n_members,H,W), amps}。
    溯源: 2025D（F04）—— 幅度摄动；乘性扰动保持量纲与非负结构。
    """
    rng = rng if rng is not None else np.random.default_rng(0)
    b = np.asarray(base, dtype=float)
    amps = rng.uniform(1 - amp_range[0], 1 + amp_range[1], size=n_members)
    members = b[None, :, :] * amps[:, None, None]
    if additive_sigma:
        members = members + rng.normal(0.0, additive_sigma,
                                       size=members.shape)
    return {"members": members, "amps": amps}


def generate_members_multimodel(predictions_list, weights=None):
    """多模型/多初值成员：把 k 个独立预报（或 k 个初值扰动同模型的输出）
    直接作为集合成员（等权或按验证期 RMSE 倒数加权）。

    数学核心: 权重 w_k ∝ 1/RMSE_k（验证期），成员不作扰动直接入集合；
    若 k 过小（<8），可对每个模型输出再叠加 perturb_scalar_members 扩容。
    参数: predictions_list [k 个 (H,W) 数组]；weights 可选 k 权重（归一化）。
    返回: dict{members: (k,H,W), weights}。
    溯源: 2025D（F04）—— 多初值/多模型成员；M6 组合预测思想的时间场延伸。
    """
    members = np.stack([np.asarray(p, dtype=float) for p in predictions_list])
    if weights is None:
        weights = np.ones(members.shape[0]) / members.shape[0]
    else:
        weights = np.asarray(weights, dtype=float)
        weights = weights / weights.sum()
    return {"members": members, "weights": weights}


def advect_member(field, flow_u, flow_v, dt, x_spacing=1.0, y_spacing=1.0):
    """半拉格朗日平流外推一步（成员级平流一致性/时间平滑用）。

    数学核心: 后向轨迹插值——场在 t+dt 的值 ≈ 场在 t 时刻沿流场回溯
      dt 后的位置插值（一阶/二阶龙格库塔回溯可选，这里给一阶）:
        x' = x − u(x,t)·dt / x_spacing,  y' = y − v(x,t)·dt / y_spacing
        field(t+dt)[x,y] ≈ interp2d(field, y', x')
    参数: field 当前场；flow_u/flow_v 流场 u/v（同形状，单位/格）；dt 步数倍数；
          x_spacing/y_spacing 格距换算（物理速度→格点数/步）。
    返回: (H,W) 平流外推场。
    用途: ①做控制预报基线；②R4 的"平流一致性检查"（外推场 vs 观测/融合场）；
          ③集合成员的物理一致性约束（成员也应满足平流方程近似）。
    溯源: 2025D（F04）—— 半拉格朗日平流；选用前先确认题面风场语义。
    """
    f = np.asarray(field, dtype=float)
    H, W = f.shape
    yy, xx = np.mgrid[0:H, 0:W]
    # 后向轨迹：t 时刻位置 = 当前 − 流场·dt
    src_x = xx - flow_u * dt / x_spacing
    src_y = yy - flow_v * dt / y_spacing
    src_x = np.clip(src_x, 0, W - 1)
    src_y = np.clip(src_y, 0, H - 1)
    fx = np.floor(src_x).astype(int)
    fy = np.floor(src_y).astype(int)
    cx = np.minimum(fx + 1, W - 1)
    cy = np.minimum(fy + 1, H - 1)
    wx = src_x - fx
    wy = src_y - fy
    out = (f[fy, fx] * (1 - wx) * (1 - wy) + f[fy, cx] * wx * (1 - wy)
           + f[cy, fx] * (1 - wx) * wy + f[cy, cx] * wx * wy)
    return out

# ============================================================================
# 二、分位聚合（P10/P50/P90 → 区间输出）
# ============================================================================

def quantile_aggregate(members, q=(10, 50, 90)):
    """逐格点分位聚合：成员场 → 分位场 + 中点 + 区间半宽。

    数学核心: P_q(x,y) = np.percentile(members[:, y, x], q)
    输出: 分位场 (len(q),H,W)、中点（P50）、区间半宽（P90−P10）/2、
          成员点均值。区间半宽即"集合离散度"（spread），
          与预测误差同量纲，可直接与 RMSE 对比做 spread-skill 检验。
    参数: members (n,H,W) 成员场；q 分位百分数序列。
    返回: dict{quantiles, q_list, median, spread_half, mean}。
    溯源: 2025D（F04）—— P10/P50/P90 输出形态。
    """
    m = np.asarray(members, dtype=float)
    qs = np.percentile(m, q, axis=0)          # (len(q),H,W)
    p10, p50, p90 = (qs[0], qs[len(q) // 2], qs[-1]) if len(q) == 3 \
        else (qs[0], np.median(m, axis=0), qs[-1])
    return {"quantiles": qs, "q_list": q,
            "median": p50, "spread_half": (p90 - p10) / 2.0,
            "mean": m.mean(axis=0)}


def forecast_interval_from_members(members, q=(10, 90)):
    """区间输出：把成员集合压缩为 (lo, hi) 区间场 + 中心，供阈值判定/航路用。

    数学核心: lo = P_q0, hi = P_q1（默认 80% 区间）；center = (lo+hi)/2。
    下游用法: 阈值判定（如湍流暴露量）用区间下/上界做稳健性双轨；
    航路规划用 center 作代价、区间半宽作风险罚项。
    返回: dict{lo, hi, center, half_width}。
    溯源: 2025D（F04）—— 区间直接服务航路稳健性决策。
    """
    m = np.asarray(members, dtype=float)
    lo, hi = np.percentile(m, q, axis=0)
    return {"lo": lo, "hi": hi, "center": (lo + hi) / 2.0,
            "half_width": (hi - lo) / 2.0}

# ============================================================================
# 三、集合可靠性检验（区间不能只是"看起来有"）
# ============================================================================

def interval_coverage(y_true, lo, hi, n_bins=10):
    """经验覆盖率 + PIT 直方图：集合区间的可靠性两件套。

    数学核心:
      覆盖率 = mean(lo ≤ y_true ≤ hi)  —— 与名义置信水平（如 80%）对比，
      显著低于名义 → 区间过窄（对不确定性低估）；显著高于 → 过宽（保守）。
      PIT = 成员分位折算的累计概率 F(y_true) 用线性插值估计，PIT 直方图均匀
      ≈ 集合校准良好；U 形=过窄、∩ 形=过宽。
    参数: y_true 真值（被降采样到与 lo/hi 同形状，或展平处理）；
          lo/hi 区间场（与 y_true 同形状）；n_bins PIT 直方图桶数。
    返回: dict{coverage, nominal_frac, pit, pit_hist}。
    溯源: 2025D（F04）—— 可靠性校准纪律；2023F（H04）—— 诚实评估。
    """
    y = np.asarray(y_true, dtype=float).ravel()
    lo = np.asarray(lo, dtype=float).ravel()
    hi = np.asarray(hi, dtype=float).ravel()
    coverage = float(np.mean((y >= lo) & (y <= hi)))
    # PIT：假设置区间端点为名义分位 (10,90)→F(lo)=0.1, F(hi)=0.9 线性插值
    f_lo, f_hi = 0.1, 0.9
    pit = np.where(y <= lo, f_lo, np.where(y >= hi, f_hi,
                                           f_lo + (y - lo) * (f_hi - f_lo) / np.maximum(hi - lo, 1e-12)))
    hist, _ = np.histogram(pit, bins=n_bins, range=(0.0, 1.0))
    return {"coverage": coverage, "nominal_frac": f_hi - f_lo,
            "pit": pit, "pit_hist": hist / max(hist.sum(), 1)}


def spread_skill_check(spread, error, corr_name="pearson"):
    """spread-skill 检验：集合离散度应与误差同步增长（好集合的自检特征）。

    数学核心: 若集合校准良好，spread² ≈ err²（平均意义），且逐样本
      corr(spread, |err|) > 0（离散度大的地方误差也大）。
    参数: spread 集合离散度（如 P90−P10）/2；error 绝对误差；
          corr_name "pearson"|"spearman"（小样本用 spearman 稳健）。
    返回: dict{corr, spread_rmse_ratio} —— ratio≈1 为理想量级。
    溯源: 2025D（F04）—— 集合形态相对单区间的可检验优势。
    """
    s = np.asarray(spread, dtype=float).ravel()
    e = np.asarray(error, dtype=float).ravel()
    if corr_name == "spearman":
        from scipy.stats import spearmanr
        r = spearmanr(s, e).statistic
    else:
        r = float(np.corrcoef(s, e)[0, 1])
    return {"corr": r,
            "spread_rmse_ratio": float(np.sqrt((s ** 2).mean()) /
                                       np.sqrt((e ** 2).mean() + 1e-12))}

# ============================================================================
# 四、平流一致性检查（R4：时空场时间维平滑的物理自检）
# ============================================================================

def advection_consistency(ens_mean_prev, ens_mean_next, flow_u, flow_v, dt,
                          x_spacing=1.0, y_spacing=1.0, metric="rmse"):
    """时间维一致性：上一时刻集合均值经平流外推后，应与下一时刻集合均值吻合。

    数学核心: residual = advect(field_t) − field_{t+1}；metric=rmse|mae|corr。
    用途: ①融合场/预报场的时间平滑是否物理（平流主导而非闪变）；
          ②判断卡尔曼递推（R4）与平流的衔接是否匹配；
          ③作为 R4"平流一致性检查"的定量输出（残差超阈值→检查流场/平滑强度）。
    返回: dict{score, residual}。
    溯源: 2025D（F04）R4 —— 融合题默认交付"空间+时间"双平滑。
    """
    pred = advect_member(np.asarray(ens_mean_prev, dtype=float),
                         np.asarray(flow_u, dtype=float),
                         np.asarray(flow_v, dtype=float),
                         dt, x_spacing, y_spacing)
    res = pred - np.asarray(ens_mean_next, dtype=float)
    r = res.ravel()
    if metric == "mae":
        score = float(np.abs(r).mean())
    elif metric == "corr":
        score = float(np.corrcoef(pred.ravel(), ens_mean_next.ravel())[0, 1])
    else:
        score = float(np.sqrt((r ** 2).mean()))
    return {"score": score, "residual": res}

# ============================================================================
# 五、装配：从一组独立预报到集合交付物（管道入口）
# ============================================================================

def build_ensemble_forecast(control_u, control_v, n_members=20,
                            amp_range=(0.10, 0.20), dir_range_deg=10.0,
                            q=(10, 50, 90), y_validate=None, rng=None):
    """集合预报一条龙：扰动 → 分位聚合 → 区间 → （有真值时）可靠性检验。

    注意: 本函数把"全场单组扰动参数"装配成最终交付物；若题面要求
    分强度档/分区域报告，请在调用侧按掩码切片后分别聚合。
    返回: dict{members_u, members_v, aggregate, interval, reliability}。
    溯源: 2025D（F04）—— 集合预报形态完整交付链。
    """
    gen = perturb_vector_members(control_u, control_v, n_members=n_members,
                                 amp_range=amp_range,
                                 dir_range_deg=dir_range_deg, rng=rng)
    speed = np.hypot(gen["members_u"], gen["members_v"])
    agg = quantile_aggregate(speed, q=q)
    ival = forecast_interval_from_members(speed, q=(q[0], q[-1]))
    out = {"members_u": gen["members_u"], "members_v": gen["members_v"],
           "amps": gen["amps"], "thetas": gen["thetas"],
           "aggregate": agg, "interval": ival}
    if y_validate is not None:
        out["reliability"] = interval_coverage(y_validate, ival["lo"], ival["hi"])
    return out