# -*- coding: utf-8 -*-
"""
image_features_modeling.py —— 图像类特征提取 + 模型 + 评估模板库（华为杯解题 Skill v0.1）
====================================================================================
【适用场景】星图识别（姿态解算）、单幅图像/视频量测（测距测高）、视频能见度估计等题目的
           "特征 → 模型 → 评估"中后段。
【溯源题号】2019B（天文导航中的星图识别）、2019C（视觉情报信息分析，交比测高）、
           2020E（能见度估计与预测，Koschmieder/暗通道）、2017D（视频显著度，问题4-6）。
【使用方法】函数骨架 + 核心数学公式注释；import 按需裁剪；输入输出 numpy 数组。
"""
import numpy as np

# ============================================================================
# 一、姿态解算数学内核（2019B 问题1 —— 三锥交汇）
# ============================================================================

def celestial_to_unit_vector(ra, dec):
    """天球坐标 (赤经 α, 赤纬 δ) → 单位方向矢量 u。

    u = (cosδ·cosα, cosδ·sinα, sinδ)ᵀ
    参数: ra, dec 可广播的数组（弧度）。
    溯源: 2019B §3.1 公式。
    """
    return np.stack([np.cos(dec) * np.cos(ra),
                     np.cos(dec) * np.sin(ra),
                     np.sin(dec)], axis=-1)


def attitude_cone_solve(ra, dec, a, f):
    """问题1(1)：f 已知时解视轴指向 d（三星三锥交汇 → 线性方程组）。

    观测方程: u_i·d = f/√(f²+a_i²) =: c_i   （d 在三个圆锥面上：锥轴 u_i、半张角 θ_i）
    线性化:   U d = c，U 的第 i 行为 u_iᵀ。
    算法:     d* = U⁻¹c（QR 或高斯消元）→ 归一化 d = d*/‖d*‖
             一致性指标 κ = |‖d*‖−1|（无噪声应为 0，天然自检量）
    退化保护: |det U| < ε 即三星近共大圆 → 构型退化（2019B 1(3) "危险构型"）。
    参数: ra, dec (3,) 弧度；a (3,) 像点距像面中心距离；f 焦距（像素）。
    返回: (alpha_D, delta_D) 视轴指向弧度的估计。n>3 时改调 attitude_least_squares。
    溯源: 2019B §3.2（算法 P1-1）。
    """
    U = celestial_to_unit_vector(ra, dec)              # (3,3)
    if abs(np.linalg.det(U)) < 1e-6:
        raise ValueError("三星共大圆，构型退化，请更换选星（2019B 1(3) 准则 1/3）")
    c = f / np.sqrt(f ** 2 + a ** 2)
    dstar = np.linalg.solve(U, c)
    kappa = abs(np.linalg.norm(dstar) - 1.0)           # 一致性指标
    d = dstar / np.linalg.norm(dstar)
    return np.arctan2(d[1], d[0]), np.arcsin(d[2]), kappa


def attitude_cone_focal_free(ra, dec, a, f_min=100.0, f_max=1e5, tol=1e-10):
    """问题1(2)：f 未知时解 d（消元 + 一维单调求根）。

    记 t = 1/f > 0，c_i(t) = (1 + a_i²t²)^(−1/2)，则 U d = c(t)。
    单位约束 ‖d‖=1 化为:  g(t) := ‖U⁻¹c(t)‖² = 1。
    关键性质（唯一性保证）: g 在 (0,∞) 严格递减（M=(U⁻¹)ᵀU⁻¹ 正定），
    故 g(0⁺)>1 时方程有且仅有一个根——二分/牛顿必收敛、无初值歧义。
    步骤: 1) 二分求 t*；2) f̂ = 1/t*；3) d = U⁻¹c(t*)（自动单位化）。
    副产品: 焦距估计 f̂ 可用于星图识别自标定（2019B §4.5）。
    若 g(0⁺)≤1（噪声/不自洽）→ 转联合非线性最小二乘（模型二，用 scipy.optimize.least_squares，
    参数化 (α_D, δ_D, log f)，初值 f₀ = a_max/tan(θ_max)）。
    溯源: 2019B §3.3（算法 P1-2 / 模型一）。
    """
    U = celestial_to_unit_vector(ra, dec)
    Ui = np.linalg.inv(U)
    M = Ui.T @ Ui

    def c(t):
        return 1.0 / np.sqrt(1.0 + a ** 2 * t ** 2)

    def g(t):
        cc = c(t)
        return cc @ M @ cc

    lo, hi = 1.0 / f_max, 1.0 / f_min
    if g(lo) <= 1.0:
        raise ValueError("g(0⁺)≤1，数据不自洽：转模型二联合最小二乘（见 docstring）")
    # 二分求根 g(t)=1（g 单调递减）
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(mid) > 1.0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    t_star = 0.5 * (lo + hi)
    f_hat = 1.0 / t_star
    d = Ui @ c(t_star)
    d = d / np.linalg.norm(d)
    return np.arctan2(d[1], d[0]), np.arcsin(d[2]), f_hat


def attitude_least_squares(ra, dec, a, f):
    """多星（n>3）最小二乘推广: d_LS = (UᵀU)⁻¹Uᵀc；带单位约束版可用拉格朗日乘子。

    理论结论（2019B 1(3) 准则5）: 全部 n 颗星整体最小二乘不劣于任何 3 星子集，
    工程上"选星"的意义 = 剔除粗差/伪星后全星参与。
    溯源: 2019B §3.2.2 推广段。
    """
    U = celestial_to_unit_vector(ra, dec)
    c = f / np.sqrt(f ** 2 + a ** 2)
    d = np.linalg.solve(U.T @ U, U.T @ c)
    d = d / np.linalg.norm(d)
    return np.arctan2(d[1], d[0]), np.arcsin(d[2])


def procrustes_rotation(u_src, w_dst):
    """正交 Procrustes：由 ≥2 对对应方向矢量恢复旋转矩阵 R（SVD 闭式解）。

    w_dst = R u_src。构造正交标架或直接 SVD:
      H = Σ_i w_i u_iᵀ = U Σ Vᵀ  →  R = V diag(1,1,det(VUᵀ)) Uᵀ
    用途: ① 像点完整坐标可用时解完整姿态（2019B §3.2.3，输出含滚转角）；
         ② 星图识别假设检验中"候选对应 → 姿态"的秒级内核（2019B 4.2 步骤7）。
    参数: u_src, w_dst: (n,3) 列向量组（n≥2，不共线）。
    溯源: 2019B §3.2.3 / §4.2。
    """
    H = w_dst.T @ u_src
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ np.diag([1, 1, np.linalg.det(Vt.T @ U.T)]) @ U.T
    return R


def pose_error_propagation(U, sigma_a, f, a, sigma_star=0.0):
    """1(3) 误差传播 + 选星评分：d 的协方差 Σ_d ≈ U⁻¹ Σ_e U⁻ᵀ。

    一阶微分: δd = U⁻¹(δc − δU·d)，
      ∂c_i/∂a_i = −f·a_i/(f²+a_i²)^{3/2}，∂c_i/∂f = a_i²/(f²+a_i²)^{3/2}
    D 角误差: σ_D² = tr[(I − ddᵀ)Σ_d]（切平面投影）。
    选星准则（实现于调用方）:
      1) 最大化 |det U|（张开度，等价最小化 cond(U)）
      2) 视场约束下边缘近等边三角形
      3) 避免像面近共线/聚拢（形状因子 最长边/最短边）
      4) 选亮星（σ_a 小）       5) n>3 全星最小二乘
    溯源: 2019B §3.4。
    """
    Ui = np.linalg.inv(U)
    dc_da = -f * a / (f ** 2 + a ** 2) ** 1.5           # 像距误差传导
    Sigma_e = np.diag((dc_da * sigma_a) ** 2 + sigma_star ** 2)
    Sigma_d = Ui @ Sigma_e @ Ui.T
    return Sigma_d


# ============================================================================
# 二、星图识别特征与算法（2019B 问题2 —— 旋转不变特征 + 假设检验）
# ============================================================================

def angular_distance_pixels(x1, y1, x2, y2, f):
    """观测星对角距（针孔模型视线夹角）。

    cos ρ = (f² + x₁x₂ + y₁y₂) / √((f²+a₁²)(f²+a₂²))，a_i² = x_i²+y_i²
    坐标必须已从像素原点换算到像面中心（去 W/2, H/2，y 轴方向待裁决）。
    溯源: 2019B §4.2 公式 (4-1)。
    """
    a1 = np.sqrt(x1 ** 2 + y1 ** 2); a2 = np.sqrt(x2 ** 2 + y2 ** 2)
    return np.arccos(np.clip((f ** 2 + x1 * x2 + y1 * y2)
                             / (np.sqrt((f ** 2 + a1 ** 2) * (f ** 2 + a2 ** 2))), -1, 1))


def catalog_pair_distance(ra1, dec1, ra2, dec2):
    """星表恒星角距（球面余弦定理）——与观测角距同单位的比对基准。

    cos γ = sinδ₁sinδ₂ + cosδ₁cosδ₂cos(α₁−α₂)
    溯源: 2019B §3.3.3 模型三公式。
    """
    return np.arccos(np.clip(np.sin(dec1) * np.sin(dec2)
                             + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2), -1, 1))


def build_star_pair_index(ra, dec, mag, gamma_fov, bin_width, mag_ok=True):
    """离线星对特征库：(角距 γ, 星等差 Δm) 双属性 + 哈希桶索引。

    设计动机（2019B 4.0 第一性原理）: 匹配特征必须是 SO(3) 不变量，
    候选只有 角距 / 星等 / 二者组合结构。双属性把单角距的哈希碰撞砍掉一个量级。
    存储节省: 只存 γ≤γ_FOV 的星对（全天球球冠占比 (1−cos γ)/2 估算规模），
    三角形闭环在线构造、不预存（对比传统三角形库省 2~3 个数量级）。
    返回: buckets: dict {(γ 桶号, Δm 桶号): [(i,j), ...]}。
    溯源: 2019B §4.1（Step 2）。
    """
    n = len(ra)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            g = catalog_pair_distance(ra[i], dec[i], ra[j], dec[j])
            if g <= gamma_fov:
                dm = (mag[i] - mag[j]) if mag_ok else 0.0
                pairs.append((i, j, g, dm))
    buckets = {}
    for i, j, g, dm in pairs:
        key = (int(g / bin_width), int(dm / bin_width))
        buckets.setdefault(key, []).append((i, j))
    return buckets


def recognize_star_pattern(obs_xy, f, catalog, buckets, tau=1.5, n_min=5):
    """单幅星图识别主流程骨架（假设—检验范式）。

    步骤: 1) 观测角距 ρ_ij 全部算出
         2) 取主星 b0（亮度优先/离群度大者）及其近邻
         3) 查哈希桶 [ρ−ε, ρ+ε] → 候选导航星对
         4) 第三星闭环校验（三边角距同时一致）→ 强假设
         5) Procrustes 解姿态 R（procrustes_rotation）
         6) 全局验证: 视场内全部导航星按 R 投影回像面，互最近邻匹配（阈值 τ·σ_a）
         7) 内点数 ≥ n_min 且投影残差 RMS < τ_R → 接受；否则换主星/放宽 ε 重试
    鲁棒性: 伪星 → 全局验证期被排除为外点；缺星 → 只要求内点数下限。
    返回: matches: [(obs_idx, catalog_id), ...]（未识别星不在其中）。
    溯源: 2019B §4.2（算法 P2）。
    """
    raise NotImplementedError(
        "骨架占位：按 docstring 的 7 步实现（哈希查询→闭环→Procrustes→全局投影验证）。\n"
        "评估指标见 evaluate_recognition()；焦距自标定: 用匹配星对代入角距公式(4-1) "
        "多对最小二乘解 f（2019B §4.5）。")


def evaluate_recognition(match_lists, true_pairs=None, ablation_levels=("A", "B", "C", "D")):
    """识别性能评估 + 消融对比统计（三支柱中的指标层）。

    自洽指标（无真值时）: 投影残差 RMS、内点比例、识别成功率、自标定 f 一致性
    仿真指标（蒙特卡洛注入噪声/伪星/缺星）: 正确识别率、误匹配率、漏匹配率、单幅耗时
    消融设计（证明"更高层次特征"价值）:
      A 纯角距哈希 → B +星等差 → C +三角形闭环 → D +姿态全局验证
      逐级报告识别率/误配率/耗时/存储，量化每层贡献。
    溯源: 2019B §4.4 三支柱评估 + 消融实验。
    """
    results = {}
    for lvl, ml in zip(ablation_levels, match_lists):
        if true_pairs is None:
            results[lvl] = {"n_img": len(ml), "status": "自洽指标待填"}
        else:
            correct = sum(1 for p in ml if p in true_pairs)
            results[lvl] = {"识别率": correct / len(true_pairs), "误配率": 1 - correct / max(len(ml), 1)}
    return results


# ============================================================================
# 三、单幅图像/视频量测（2019C —— 交比测高 / 地面距离）
# ============================================================================

def cross_ratio_height(v, m, b, t, h_c):
    """交比测高：竖直线上一维射影保持交比，由像点反推真实高度 H。

    公式: H = h_c · (1 − CR(v,m; b,t))，
          CR(v,m;b,t) = (b−v)(t−m) / ((b−m)(t−v))
    其中 v=铅垂灭点像，m=竖直线与地平线交点像（恰为"与相机同高"点），
    b=竖直段底部（地面 z=0），t=顶部；沿竖直线取一致有向像素坐标。
    反解相机高度（已知参照段 H_ref）: h_c = H_ref / (1 − CR(v,m; b_ref,t_ref))。
    验证链: t→b 时 CR→1, H→0 ✓；t→m 时 CR→0, H→h_c ✓。
    溯源: 2019C §3.2 ② 公式 (H)(H′)。
    """

    def cr(p1, p2, p3, p4):
        return ((p3 - p1) * (p4 - p2)) / ((p3 - p2) * (p4 - p1))
    return h_c * (1.0 - cr(v, m, b, t))


def ground_distance_from_pitch(h_cam, px, py, K, v_z, y_h=None):
    """地面点水平距离（深度公式族): D = −h·cotα。

    cos α = (K⁻¹p̃)·(K⁻¹ṽ_z) / (‖K⁻¹p̃‖·‖K⁻¹ṽ_z‖)，α∈(0,π) 为视线与铅垂夹角
    D_P = −h·cotα = −h·cosα / √(1−cos²α)
    特例（光轴近水平无滚转）: D ≈ h·f/(y_p − y_h)，"像素纵向差反比定距"。
    误差结构: σ_D ≈ D²σ(Δy)/(h·f) —— 距离越远误差按平方增长，远距给置信区间。
    溯源: 2019C §3.2 ① 公式 (D)。
    """
    Ki = np.linalg.inv(K)
    p = Ki @ np.array([px, py, 1.0])
    vz = Ki @ v_z
    cos_a = (p @ vz) / (np.linalg.norm(p) * np.linalg.norm(vz))
    if y_h is not None:                      # 特例快速估算（可选）
        return h_cam * K[0, 0] / (py - y_h)
    return -h_cam * cos_a / np.sqrt(max(1 - cos_a ** 2, 1e-12))


# ============================================================================
# 四、能见度估计（2020E —— Koschmieder / 暗通道先验）
# ============================================================================

def koschmieder_visibility(C0, C_d, d, eps=0.05):
    """对比度衰减法：由目标固有对比度与观测对比度反解消光系数与 MOR。

    物理: C(d) = C₀·e^(−σd)、MOR = ln(1/ε)/σ（ε=0.05 气象光学视程口径 ≈ 2.996/σ；
         航空 RVR 用 ε=0.02 → 3.912/σ，论文须统一口径并说明换算）。
    参数: C0 固有对比度（目标与背景亮度差/背景），C_d 观测对比度，d 物理景深(m)。
    返回: (σ, MOR)。多条标线/多目标可做加权多观测最小二乘。
    溯源: 2020E §3.9 对比度衰减 + 假设 A8。
    """
    sigma = -np.log(max(C_d / C0, 1e-12)) / max(d, 1e-6)
    return sigma, np.log(1.0 / eps) / sigma


def dark_channel_transmission(img, A_est=None, omega=0.95):
    """暗通道先验估计透射率图 t(x)（大气光反演算法的核心块）。

    模型: I(x) = J(x)·t(x) + A∞(1 − t(x))，t(x) = e^(−σ·d(x))
    暗通道: J_dark(x) = min_{y∈Ω(x)} min_{c∈{r,g,b}} J_c(y) → 0（无雾先验）
    透射率: t(x) = 1 − ω·min_y(min_c I_c(y)/A_c)（ω≈0.95 保远景雾感）
    A∞: 取暗通道最亮 0.1% 像素在原图中的最大亮度。
    用途: Q3 算法B（备），需清晰参考帧或全局大气光；得 σ 后换算 MOR。
    溯源: 2020E §3.11 算法B + §3.6 物理辅助分支（透射率头）。
    """
    from scipy.ndimage import minimum_filter
    if A_est is None:
        dark = np.min(minimum_filter(img, size=15), axis=-1)
        A_est = img[np.unravel_index(np.argmax(dark), dark.shape)].max()
    dark_c = np.min(minimum_filter(img / np.maximum(A_est, 1e-6), size=15), axis=-1)
    t = 1.0 - omega * dark_c
    return np.clip(t, 0.1, 1.0), A_est


# ============================================================================
# 五、评估通用件（2017D 问题4/6、2019B、2020E）
# ============================================================================

def saliency_time_series(S, masks):
    """逐帧显著度指标（2017D 问题4 决策层输入）。

    s1(t) = 前景像素占比；s2(t) = 最大连通域面积占比；s3(t) = Σ|S_(:,t)|
    综合显著度 s(t) = s1^α · s2^(1−α)（α≈0.5）或 s2 为主。
    自适应阈值: 取 s(t) 低分位（10%）鲁棒基线 μ0=MAD 中位数，θ = μ0 + k·σ0 (k≈3~5)。
    时序事件化: 超阈值连通段合并（容忍 1~2 帧间隙、丢弃 <3~5 帧短段、边界扩展 1 帧）。
    溯源: 2017D §3 问题4 第二步~第四步。
    """
    s2 = np.array([m.max(axis=0).sum() if m.ndim == 3 else 0.0 for m in masks])
    if isinstance(S, np.ndarray):
        s3 = np.abs(S).sum(axis=0)
    else:
        s3 = np.zeros(len(masks))
    s1 = np.array([m.mean() for m in masks])
    return s1, s2 / max(s1.sum() or 1, 1), s3


def anomaly_score_mahalanobis(Z, mu=None, Sigma=None):
    """异常检测的马氏距离打分（正常基线偏离检测）。

    a_t = (z_t − μ)ᵀΣ⁻¹(z_t − μ)，超 χ²_d 分位报警；渐变型用 CUSUM 累积。
    基线须用鲁棒统计（中位数/MAD 或截尾估计）防正常段混入轻微异常。
    溯源: 2017D §3 问题6 模型层 2。
    """
    if mu is None:
        mu = np.median(Z, axis=0)
    if Sigma is None:
        from scipy.stats import median_abs_deviation
        Sigma = np.diag(median_abs_deviation(Z, axis=0) * 1.4826) ** 2
    d = Z - mu
    return np.einsum('ij,jk,ik->i', d, np.linalg.inv(Sigma + 1e-6 * np.eye(len(mu))), d)