# -*- coding: utf-8 -*-
"""
uncertainty_optimization.py —— 不确定性分析与约束优化模板库（华为杯解题 Skill v0.1）
==================================================================================
【适用场景】"预测之后还要找最优解/给区间"的收尾题：反向设计（2021D 问题4）、
           操作条件优化（2020B 问题4/5）、误差与灵敏度分析（2019B 1(3)、2018D）、
           外推预测与消散时刻（2020E 问题4）、极值统计（2019E 问题3）。
【溯源题号】2021D（代理模型 + PSO/GA + 分位数可行域 + OAT/Sobol 敏感性，问题4）、
           2020B（LP/QP + 差分进化 + ε-约束帕累托 + 逐步路径 + 蒙特卡洛复核，
           问题4/5）、2018D（交叉点一致性 + 复差指标 + 误差传播）、2019B（选星
           误差传播 Σ_d = U⁻¹Σ_eU⁻ᵀ）、2019E（GEV 极值统计）、2020E（外推置信区间）。
【核心思想】模型即代理（surrogate），约束 = 数据支撑的可行域；一切"达标判定"
           都要带误差容差与稳健性复核（2020B A9/A10、2021D "第二种方法复核"）。
"""
import numpy as np

# ============================================================================
# 一、可行域定界（2021D 问题4 / 2020B 问题4 —— 防代理外推失真的前提）
# ============================================================================

def feasible_box_quantile(X, lo_q=0.05, hi_q=0.95):
    """数据驱动可行域: Ω = Π_d [q5%, q95%]（描述符/操作变量分位箱）。

    动机: 数据驱动模型外推不可靠，把搜索限制在有数据支撑的区间内
    （2021D 独立解答 207 行：training 集 5%-95% 分位；获奖论文可扩张约 20%）。
    若题目给硬范围（2020B 附件四操作范围 [L_j, U_j]）→ 取 分位箱 ∩ 硬范围。
    溯源: 2021D §Q4 可行域 + 2020B A7。
    """
    return (np.quantile(X, lo_q, axis=0), np.quantile(X, hi_q, axis=0))


# ============================================================================
# 二、代理模型 + 约束优化（2020B 问题4 线性族 / 2021D 问题4 黑箱族）
# ============================================================================

def lp_qp_optimize_objective(theta, X_fixed, hl, hs):
    """线性/二次代理的约束优化（LP/QP 精确求解路径，2020B Q4 工程红利）。

    min_x L̂(x)  s.t.  Ŝ(x) ≤ 5 μg/g（硫约束）、L_j ≤ x_j ≤ U_j、不可调变量固定。
    说明: 若 L̂、Ŝ 为线性 → LP；含二次项 → QP。scipy.optimize.linprog /
    minimize(method='SLSQP') 或 CVXPY 均可；几十变量毫秒级，逐样本 325 次无压力。
    达标判定: (L_obs − L̂(x*))/L_obs > 0.30，基线用样本自身实测（A10）；
    不可行时报告可达最大降幅 + 激活约束瓶颈（硫约束？边界？）。
    溯源: 2020B §7.1-7.3 全节。
    """
    raise NotImplementedError(
        "骨架占位：构造 x = [可调操作变量]，目标 = L̂(x)（线性系数 θ_L），\n"
        "约束 = Ŝ(x)−5 ≤ 0（线性系数 θ_S）+ 上下界；逐样本循环求解。\n"
        "f(x) 为线性: scipy.optimize.linprog；二次: minimize+SLSQP 或 CVXPY。")


def heuristic_optimize_constrained(f_surrogate, x0, bounds, g_constraints=None,
                                   method="de", n_restarts=10, seed=0):
    """黑箱代理 + 启发式优化（差分进化 / PSO / 遗传算法）—— 约束用罚函数/修复算子。

    适用于 L̂/Ŝ 为 RF/XGB 等无梯度模型的场景（2020B §7.2 / 2021D 对齐点12）。
    做法: ① 多起始点多方法（DE + PSO + GA）取稳定最优，防局部极小；
         ② 约束处理: 罚函数 f + big_M·Σmax(0, g_i) 或修复算子；
         ③ 收敛后对最优解做邻域网格验证 + 蒙特卡洛复核（见后）。
    遗传算法细节（2021D 独立解答 212 行）: 实数编码、锦标赛选择、SBX 交叉、
     多项式变异 —— 可直接复用 scipy differential_evolution 或手写 GA 骨架。
    溯源: 2021D §Q4 启发式优化 + 2020B §7.2。
    """
    from scipy.optimize import differential_evolution

    def penalized(x):
        val = f_surrogate(x)
        if g_constraints:
            val += 1e6 * sum(max(0.0, g(x)) for g in g_constraints)
        return val

    best_x, best_f = None, np.inf
    for s in range(n_restarts):
        res = differential_evolution(penalized, bounds, seed=seed + s,
                                     tol=1e-6, polish=True)
        if res.fun < best_f:
            best_f, best_x = res.fun, res.x
    return best_x, best_f


def pso_optimize(f_surrogate, bounds, n_particles=40, n_iter=200, seed=0):
    """PSO 骨架（粒子群，numpy 手写版 —— scipy 无内置，20 行内可落地）。

    v ← w·v + c1·r1·(p_best − x) + c2·r2·(g_best − x)；x ← x + v
    惯性 w 线性衰减 (0.9→0.4)，约束同样用罚函数。获奖论文"PSO 多目标优化 +
    Pareto 解集 + 主要目标法验证"（2021D 对齐点12 武大做法）即在此骨架上加
    多目标聚合。溯源: 2021D 对齐点12（武大 PSO）、独立解答（PSO 具名）。
    """
    n_dim = len(bounds)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    x = np.random.RandomState(seed).uniform(lo, hi, (n_particles, n_dim))
    v = np.zeros_like(x)
    p_best, p_f = x.copy(), np.array([f_surrogate(xi) for xi in x])
    g_best = p_best[np.argmin(p_f)]
    w0, w1, c1, c2 = 0.9, 0.4, 2.0, 2.0
    for it in range(n_iter):
        w = w0 - (w0 - w1) * it / n_iter
        r1, r2 = np.random.rand(n_particles, n_dim), np.random.rand(n_particles, n_dim)
        v = w * v + c1 * r1 * (p_best - x) + c2 * r2 * (g_best - x)
        x = np.clip(x + v, lo, hi)
        f_cur = np.array([f_surrogate(xi) for xi in x])
        better = f_cur < p_f
        p_best[better], p_f[better] = x[better], f_cur[better]
        if p_f.min() < f_surrogate(g_best):
            g_best = p_best[np.argmin(p_f)]
    return g_best, p_f.min()


def pareto_frontier_epsilon_constraint(f_obj, f_con, grid_values, bounds, method="de"):
    """ε-约束法帕累托前沿: 固定约束上限扫网格，逐点求目标最小（损失-硫权衡图）。

    物理背景（2020B A3）: 脱硫深度与 RON 损失此消彼长 → 帕累托前沿展示
    "5 μg/g 硫约束的机会成本"。逐点将约束转为 g(x) ≤ ε 后调启发式优化器。
    溯源: 2020B §7.2 可选加分（ε-约束法 4/5/6 μg/g 扫描）+ 2021D Pareto 解集。
    """
    pts = []
    for eps in grid_values:
        x, f = heuristic_optimize_constrained(
            f_obj, None, bounds, g_constraints=[lambda xi, e=eps: f_con(xi) - e])
        pts.append((x, f, eps))
    return pts


# ============================================================================
# 三、灵敏度分析（2021D OAT + Sobol；2019B 选星误差传播；2018D 交叉点检验）
# ============================================================================

def oat_sensitivity(f, x_nom, bounds, n_grid=30):
    """OAT 单因素敏感性: 其余变量固定训练均值，逐个扫描 [l_d, u_d]。

    输出 "描述符取值 → 目标" 曲线；与 ADMET 达标曲线求交集 = 推荐区间。
    每个入选变量汇报三数: 活性递增方向 / 达标区间 / 二者交集（2021D 输出约定）。
    溯源: 2021D 独立解答 215 行（OAT 单因素敏感性分析 + 区间落地）。
    """
    curves = {}
    for d in range(len(bounds)):
        grid = np.linspace(bounds[d][0], bounds[d][1], n_grid)
        vals = []
        for gv in grid:
            x = np.array([np.clip(x_nom[i], *bounds[i]) for i in range(len(bounds))])
            x[d] = gv
            vals.append(f(x))
        curves[d] = (grid, np.asarray(vals))
    return curves


def sobol_first_order_sensitivity(f, bounds, n_samples=2048, seed=0):
    """Sobol 全局灵敏度（一阶 Si + 总效应 STi）—— Saltelli 采样 + 方差分解。

    数学: 对独立均匀输入，Saltelli 矩阵 A, B, AB_i (n×(2d+2))；
      Si = Var(E[f|X_i])/Var(f) ≈ [mean(f_B·(f_ABi − f_A))]/Var(f)
      STi = 1 − Var(E[f|X~i])/Var(f) ≈ [mean(f_A·(f_A − f_ABi))]/Var(f)
    （骨架版手写 Saltelli + Monte Carlo 估计；工程上可直接 pip install SALib，
     其 saltelli.sample + sobol.analyze 与此公式等价。）
    用途: 描述符重要性排序的全局视角 —— 与 OAT 互补（OAT 快、局部；Sobol 全局）。
    溯源: 2021D 差距分析 line 20（"Sobol 敏感性"被列为实操要点）+ 独立解答 Q2/Q4。
    """
    rng = np.random.RandomState(seed)
    d = len(bounds)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    A = rng.uniform(0, 1, (n_samples, d)) * (hi - lo) + lo
    B = rng.uniform(0, 1, (n_samples, d)) * (hi - lo) + lo
    fA = np.array([f(a) for a in A]); fB = np.array([f(b) for b in B])
    var_f = np.var(np.concatenate([fA, fB]))
    if var_f < 1e-12:
        return np.zeros(d), np.zeros(d)
    Si, STi = [], []
    for i in range(d):
        ABi = B.copy(); ABi[:, i] = A[:, i]
        fABi = np.array([f(a) for a in ABi])
        Si.append(np.mean(fB * (fABi - fA)) / var_f)
        BAi = A.copy(); BAi[:, i] = B[:, i]
        fBAi = np.array([f(a) for a in BAi])
        STi.append(1.0 - np.mean(fA * (fA - fBAi)) / var_f)
    return np.asarray(Si), np.asarray(STi)


# ============================================================================
# 四、预测区间与不确定性传播（2020E / 2018D / 2019B / 2019E）
# ============================================================================

def residual_interval_prediction(y_pred, resid, alpha=0.05, hetero_bins=None):
    """残差驱动的预测区间: 均值预测 ± 分位数带（正态近似或分位数回归）。

    做法: ① 残差正态近似: σ̂ = MAD(resid)·1.4826，CI = ŷ ± z_{1−α/2}·σ̂；
         ② 稳健版（重尾）: 直接取残差分位数 [q_{α/2}, q_{1−α/2}] 作加性带；
         ③ 分层版（2020E 评估分层思想）: 按预测值分档（<400/400-800/...）每档
            独立估计残差尺度 —— 低能见度档要求误差最小（业务关键区）。
    用途: 2020E Q4 消散时刻 t*（MOR=150m 穿越点）必须给置信区间而非点值；
         2020B A9 达标判定设置测量误差容差带。
    溯源: 2020E §3.13/§3.17；2020B A9/A10。
    """
    if hetero_bins is None:
        lo, hi = np.quantile(resid, [alpha / 2, 1 - alpha / 2])
        return y_pred + lo, y_pred + hi
    bands = np.zeros((len(y_pred), 2))
    for k in range(len(hetero_bins) - 1):
        m = (y_pred >= hetero_bins[k]) & (y_pred < hetero_bins[k + 1])
        bands[m] = np.quantile(resid[m], [alpha / 2, 1 - alpha / 2]) if m.any() else 0
    return y_pred + bands[:, 0], y_pred + bands[:, 1]


def monte_carlo_robustness_check(x_star, f_obj, bounds, n_draws=500, pert_frac=0.02, seed=0):
    """最优解稳健性复核: 对 x* 局部蒙特卡洛扰动，确认达标结论不敏感。

    （2020B §7.3 步5）: 操作执行误差 ±小量、模型参数重抽样（bootstrap）;
    统计扰动域内目标达标率（如硫≤5 且降幅>30% 的比例）。
    2021D 对齐点12 的"用第二种方法复核解"亦属此精神（PSO 解 → 主要目标法复核）。
    溯源: 2020B §7.3 步5 + A9 容差带；2021D 对齐点12。
    """
    rng = np.random.RandomState(seed)
    ok = 0
    for _ in range(n_draws):
        xp = np.clip(x_star * (1 + rng.uniform(-pert_frac, pert_frac, len(x_star))),
                     [b[0] for b in bounds], [b[1] for b in bounds])
        ok += int(f_obj(xp))
    return ok / n_draws


def stepwise_feasible_path(x0, x_star, delta_max, bounds, f_dual=None, thresh=None):
    """逐步调整可行路径（2020B 问题5）: 每步每变量最多 Δ，全程满足约束。

    方案 P1（并行渐近）: x_j⁽ᵗ⁾ = clip(x_j⁽ᵗ⁻¹⁾ + sign(x*_j − x_j⁽ᵗ⁻¹⁾)·
      min(|Δ|, Δ_j), L_j, U_j)；步数 = max_j ⌈|x*_j − x⁰_j|/Δ_j⌉。
    方案 P2（贪心序贯）: 每步选"移动 Δ 后目标最优且约束满足"的变量执行。
    可行修正: P1 中途若硫预测超限 → 该步改 P2 绕行（先动降硫变量）。
    验证三件套: 终点=求解器最优（容差内）/ 全程约束满足（逐行断言）/ 步幅合规。
    溯源: 2020B §8.1-8.3（P1/P2 + 验证方法）。
    """
    x = np.asarray(x0, dtype=float); xs = np.asarray(x_star, dtype=float)
    dm = np.asarray(delta_max, dtype=float)
    lo = np.array([b[0] for b in bounds]); hi = np.array([b[1] for b in bounds])
    path = [x.copy()]
    while np.max(np.abs(x - xs) / np.maximum(dm, 1e-9)) > 1e-6:
        step = np.sign(xs - x) * np.minimum(np.abs(xs - x), dm)
        x = np.clip(x + step, lo, hi)
        if f_dual is not None and f_dual(x) > thresh:   # 中途约束破坏 → P2 绕行挂点
            pass
        path.append(x.copy())
        if len(path) > 1000:
            break
    return np.asarray(path)


def vector_difference_error(H_a, g_a, H_b, g_b, wrap=180.0):
    """复差指标: 两套调和常数（复平面矢量）的差异 D（比单看 ΔH/Δg 更公平）。

    D = √(H_a² + H_b² − 2H_aH_b·cos(Δg))，相对误差 D/H_b（避免小振幅区虚高）。
    迟角差缠绕到 [−180°, 180°]（g 跨 0°/360° 跳变必须先 wrap）。
    用途: 2018D 问题1 验潮站外部检验 + 问题2 交叉点一致性 RMS（获奖论文同款指标）。
    溯源: 2018D §4.5 步2 复差 + 对齐点 A5/A6（矢量差 RMS 交叉点检验）。
    """
    dg = np.deg2rad(((g_a - g_b + wrap) % (2 * wrap)) - wrap)
    D = np.sqrt(H_a ** 2 + H_b ** 2 - 2 * H_a * H_b * np.cos(dg))
    return D, D / np.maximum(H_b, 1e-12)


def extreme_value_quantile(sample, return_period_years, block=365):
    """极值统计: 年极值分块 + GEV 拟合，重现期分位（2019E 问题3 尾部论证）。

    思路: 极端天气频率论证 = 均值上升 + 尾部分布（GEV 位置/尺度参数变化）。
    实现: 分块取年极值 → scipy.stats.genextreme 拟合 → 重现期分位
          x_T 满足 1 − F(x_T) = 1/T（或对尾指数用 Pareto 拟合）。
    溯源: 2019E §三 问题3 统计模型（分布漂移 + 极值统计）。
    """
    from scipy.stats import genextreme
    n_blocks = len(sample) // block
    block_max = np.array([sample[i * block:(i + 1) * block].max() for i in range(n_blocks)])
    c, loc, scale = genextreme.fit(block_max)
    p = 1.0 - 1.0 / return_period_years
    return genextreme.ppf(p, c, loc=loc, scale=scale), (c, loc, scale)