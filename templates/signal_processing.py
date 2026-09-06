# -*- coding: utf-8 -*-
"""
signal_processing.py —— 阵列/谱估计模板库（华为杯解题 Skill v0.9）
==================================================================================
【适用场景】多通道/多传感器信号的多谐波与多参数估计："定阶 → 子空间超分辨 →
           相干解相干 → 幅相误差自校准 → 稀疏重构"完整主链路；适用于传感器阵列/
           导波/工频谐波/多普勒谱类谱估计题（2022A 谱估计与参数辨识环节）。
【溯源题号】2022A（H01）。
【核心纪律】① 阶数决定子空间维数，先用 MDL/AIC 定阶，禁止拍脑袋定信源数；
           ② 子空间法要求信源数 < 阵元数，且相干源必须先解相干再进 MUSIC
           （否则秩亏，谱峰消失）；③ 一律用 SVD（np.linalg.svd）取噪声子空间，
           不显式构造 X·Xᴴ（数值更稳）；④ 谱峰搜索网格密度 = 精度上限，
           粗网格 + 抛物线三点插值细化两段式。
"""
import numpy as np

# ============================================================================
# 一、模型阶数估计（MDL/AIC）—— 一切子空间法的前置
# ============================================================================

def estimate_order_mdl_aic(X, max_order=None):
    """MDL / AIC 信息论定阶：从样本协方差特征谱决定信源个数。

    数学核心（Wax-Kailath 准则，K = 候选信源数，M = 通道数，N = 快拍数）:
      L(K)      = N·(M−K)·log( λ_arith / λ_geo )     λ_arith/λ_geo 为最小
                   M−K 个特征值的算术/几何均值（arith ≥ geo，L ≥ 0）
      AIC(K)    = 2·L(K) + 2·K·(2M−K)      （推导: −2ℓ(K) 展开后常数项
                                              N·Σln λ_i 跨 K 抵消，剩 2·L(K)）
      MDL(K)    = L(K) + 0.5·K·(2M−K)·log(N)
    取使准则最小的 K ∈ [0, max_order]。特征值取 R = X·Xᴴ/N，用奇异值²/N 计算。
    注意: AIC 渐近不一致（大 N 时会过估阶数），MDL 一致 —— 双准则冲突时
    以 MDL 为准、AIC 作参考，论文把两值都报告。
    参数: X (M×N 复数/实数观测矩阵，M 通道 × N 快拍)；max_order 候选上限
          （默认 M−1）。
    返回: dict{AIC_best, MDL_best, eigvals} —— 双准则结论 + 特征谱（画图诊断）。
    适用场景: 谐波/信号源个数未知、需要先定阶再进 MUSIC/ESPRIT 的环节。
    溯源: 2022A（H01）—— 定阶是子空间法的第一道工序。
    """
    M, N = X.shape
    max_order = M - 1 if max_order is None else min(max_order, M - 1)
    _, s, _ = np.linalg.svd(X)
    lamb = np.maximum((s ** 2) / N, 1e-12)             # 特征值 = 奇异值²/N，更稳
    aic, mdl = [], []
    for k in range(0, max_order + 1):
        small = lamb[k:]
        n_small = M - k
        arith = small.mean()
        geo = np.exp(np.mean(np.log(small)))
        L = N * n_small * np.log(arith / geo)
        aic.append(2.0 * L + 2.0 * k * (2 * M - k))
        mdl.append(L + 0.5 * k * (2 * M - k) * np.log(N))
    return {"AIC_best": int(np.argmin(aic)), "MDL_best": int(np.argmin(mdl)),
            "eigvals": lamb}


# ============================================================================
# 二、子空间超分辨：MUSIC / ESPRIT / root-MUSIC
# ============================================================================

def _steering(theta_grid, M, d_over_lambda):
    """ULA 均匀线阵导向矢量矩阵: a(θ) = [1, e^{jφ}, …, e^{j(M−1)φ}]ᵀ，
    φ = 2π·(d/λ)·sinθ。返回 (M, L)，每列对应网格上一个角度（θ 单位为度）。"""
    theta_grid = np.atleast_1d(np.asarray(theta_grid, dtype=float))
    phi = 2.0 * np.pi * d_over_lambda * np.sin(np.radians(theta_grid))
    return np.exp(1j * np.outer(np.arange(M), phi))    # (M, L)


def music_from_cov(R, n_sources, theta_grid, M, d_over_lambda=0.5):
    """MUSIC 伪谱（协方差入口）: P(θ) = 1 / ‖a(θ)ᴴ·U_n‖²，峰位即角度估计。

    数学核心（子空间正交性）: 信号子空间与噪声子空间正交，导向矢量 a(θ) 当且仅
    当 θ 等于真实角度时落在信号子空间内 ⇒ ‖U_nᴴ·a(θ)‖² → 0 ⇒ 伪谱 → ∞。
    步骤: ① R 的 SVD 取后 (M−n_sources) 个特征向量为噪声子空间 U_n；
          ② 网格 θ 上算 P(θ)；③ 取峰（配 spectrum_peaks 抛物线插值细化）。
    参数: R (M×M 协方差，可直接传解相干后的平滑协方差)；theta_grid（度）。
    返回: 伪谱数组（与 theta_grid 等长），谱峰坐标用 spectrum_peaks 提取。
    适用场景: 通道数充足、信源互不相关时的角度/频率估计主路径。
    溯源: 2022A（H01）。
    """
    M_r = R.shape[0]
    if n_sources >= M_r:
        raise ValueError("信源数必须小于阵元数（子空间法前提），先定阶再调用")
    _, _, vh = np.linalg.svd(R)
    Un = vh[n_sources:].conj().T                        # (M, M−K) 噪声子空间
    A = _steering(theta_grid, M_r, d_over_lambda)       # (M, L)
    denom = np.sum(np.abs(Un.conj().T @ A) ** 2, axis=0)
    return 1.0 / np.maximum(denom, 1e-12)


def music_spectrum(X, n_sources, theta_grid, d_over_lambda=0.5):
    """MUSIC 伪谱（观测数据入口）: 自动构造协方差后走 music_from_cov。"""
    M, N = X.shape
    R = X @ X.conj().T / N
    return music_from_cov(R, n_sources, theta_grid, M, d_over_lambda)


def spectrum_peaks(spectrum, theta_grid, n_peaks, refine=True):
    """伪谱取峰: 局部极大粗检 + 三点抛物线插值细化（亚网格精度）。

    数学核心（抛物线顶点公式）: 峰位 x* = x_i + 0.5·(s_{i−1}−s_{i+1}) /
    (s_{i−1}−2s_i+s_{i+1})·Δθ —— 一阶差分置零解出，网格再密也不如这步划算。
    返回: [(θ_峰, 峰高), …] 按峰高降序，取前 n_peaks 个。
    溯源: 2022A（H01）—— 谱搜索环节的标准收尾。
    """
    s = np.asarray(spectrum, dtype=float)
    g = np.asarray(theta_grid, dtype=float)
    idx = np.where((s[1:-1] > s[:-2]) & (s[1:-1] > s[2:]))[0] + 1   # 局部极大
    idx = idx[np.argsort(s[idx])[::-1]][:n_peaks]
    peaks = []
    for i in idx:
        g_peak = float(g[i])
        if refine and 0 < i < len(s) - 1:
            d = s[i - 1] - 2.0 * s[i] + s[i + 1]
            if abs(d) > 1e-12:
                g_peak = float(g[i] + 0.5 * (s[i - 1] - s[i + 1]) / d * (g[i + 1] - g[i]))
        peaks.append((g_peak, float(s[i])))
    return peaks


def esprit_estimate(X, n_sources, d_over_lambda=0.5):
    """LS-ESPRIT 闭式角度估计（免谱搜索）: Ψ = (U₁ᴴU₁)⁻¹U₁ᴴU₂ 特征值取相位。

    数学核心（旋转不变性）: 信号子空间 U_s 去掉末行/首行得 U₁/U₂，二者经
    子阵偏移满足 U₂ ≈ U₁·Ψ；Ψ 的特征值 λ_k 携带角度信息:
      arg(λ_k) = 2π·(d/λ)·sinθ_k  ⇒  θ_k = arcsin( arg(λ_k) / (2π·d/λ) )
    步骤: ① SVD 取前 n_sources 列 U_s；② 切 U₁=U_s[:-1], U₂=U_s[1:]；
          ③ Ψ = lstsq(U₁, U₂)；④ 角度 = arcsin(angle(eig(Ψ))/(2π d/λ))。
    返回: 角度估计数组（度）。TLS-ESPRIT（对噪声更稳）: 对 [U₁|U₂] 做 SVD、
    取右奇异向量的右下块做 Ψ_TLS = −V₁₂·V₂₂⁻¹，其余相同。
    适用场景: 角度/频率个数已知、要闭式解写进论文的环节（与 MUSIC 互为对照）。
    溯源: 2022A（H01）。
    """
    M, _N = X.shape
    if n_sources >= M:
        raise ValueError("信源数必须小于阵元数（子空间法前提）")
    U, _, _ = np.linalg.svd(X, full_matrices=False)
    Us = U[:, :n_sources]
    U1, U2 = Us[:-1, :], Us[1:, :]
    Psi = np.linalg.lstsq(U1, U2, rcond=None)[0]        # (U₁ᴴU₁)⁻¹U₁ᴴU₂
    w = np.angle(np.linalg.eigvals(Psi))
    w = np.sort(w)[::-1]
    theta = np.degrees(np.arcsin(np.clip(w / (2.0 * np.pi * d_over_lambda), -1.0, 1.0)))
    return theta


def root_music(X, n_sources, d_over_lambda=0.5):
    """root-MUSIC: 谱搜索 → 多项式求根，同一信息量下精度更高、无网格伪影。

    数学核心: 在单位圆 z = e^{jφ}（φ = 2π(d/λ)sinθ）上让伪谱分母多项式化。
    相位约定（易错，务必与 MUSIC 的导向矢量口径一致）: 按
      D(z) = z^{M−1} · p(1/z)ᵀ · (U_n U_nᴴ) · p(z)   （p(z) = [1, z, …, z^{M−1}]ᵀ）
    组装时，单位圆上 D(e^{jφ}) ∝ ‖a(φ)ᴴU_n‖²，零点恰在 φ = +φ_true ——
    根相位直接对应真实角度；若误用 p(z)ᵀ·C·p(1/z) 组装，全部根会整体反号
    （角度取负、符号审查时易踩坑）。展开为 2M−2 次多项式:
      系数 c_k = Σ_{n−m=k−(M−1)} [U_nU_nᴴ]_{m,n}
    选根要点: 根成对 {z, 1/conj(z)}（同相位、半径互反），每个信源恰有一个
    单位圆内根 —— 先在 |z|<1 的根里取模最接近 1 的 n_sources 个，否则会误取
    镜像根导致角度重复。
    步骤: ① SVD 求噪声子空间；② 按上式组装多项式系数（np.roots 系数顺序
          = 从最高次到常数项）；③ 单位圆内挑根 + 换算角度。
    适用场景: 与 MUSIC 双路线对照论证（同一协方差，两种峰提取口径一致即互证）。
    溯源: 2022A（H01）。
    """
    M, N = X.shape
    if n_sources >= M:
        raise ValueError("信源数必须小于阵元数（子空间法前提）")
    R = X @ X.conj().T / N
    _, _, vh = np.linalg.svd(R)
    C = vh[n_sources:].conj().T @ vh[n_sources:]        # U_n·U_nᴴ (M×M)
    coef = np.zeros(2 * M - 1, dtype=complex)           # 下标: 次数 → 位置
    for m in range(M):
        for n2 in range(M):
            pos = (2 * M - 2) - ((M - 1) + (n2 - m))    # p(1/z)ᵀC p(z): 幂=M−1−m+n
            coef[pos] += C[m, n2]
    roots = np.roots(coef)
    inside = roots[np.abs(roots) < 1.0]                 # 根成对 {z,1/z}，内/外各一
    if inside.size < n_sources:                         # 数值退化兜底: 放宽半径
        inside = roots[np.argsort(np.abs(np.abs(roots) - 1.0))[:2 * n_sources]]
    dist = np.abs(np.abs(inside) - 1.0)                 # 到单位圆距离
    pick = inside[np.argsort(dist)[:n_sources]]
    w = np.angle(pick)
    w = np.sort(w)[::-1]
    theta = np.degrees(np.arcsin(np.clip(w / (2.0 * np.pi * d_over_lambda), -1.0, 1.0)))
    return theta


# ============================================================================
# 三、相干信号解相干（前后向平滑 / Toeplitz）—— MUSIC 前置必做
# ============================================================================

def spatial_smoothing(X, subarray_len):
    """前向空间平滑: 相干多径信号去相关，恢复协方差满秩。

    数学核心: 相干源使 R 秩亏（秩 < 信源数），子空间正交性失效。把 M 元阵切成
    L = M − p + 1 个长度 p 的重叠子阵，平均子阵协方差:
      R_ss = (1/L)·Σₗ Rₗ      （Rₗ 取原协方差 R 的 [l:l+p, l:l+p] 块）
    平滑后信源数上限升到 p（前向平滑可解 p 个相干源）。
    返回: (R_ss, p) —— 平滑协方差 + 子阵长度；后续 MUSIC 用 music_from_cov
          且阵元数参数传 p（导向矢量维度随之变化）。
    溯源: 2022A（H01）—— 多径/多途相干场景的标准预处理。
    """
    M, N = X.shape
    p = min(int(subarray_len), M)
    L = M - p + 1
    R = X @ X.conj().T / N
    R_sum = np.zeros((p, p), dtype=complex)
    for l in range(L):
        R_sum += R[l:l + p, l:l + p]
    return R_sum / L, p


def forward_backward_smoothing(X, subarray_len):
    """前后向空间平滑 FBSS: 前向 + 共轭倒序平滑取平均，去相干能力翻倍。

    数学核心（后向平滑 = 对数据共轭反转后再前向平滑，数学上等价）:
      R_fb = (R_f + J·R_f*·J) / 2，J 为 p×p 反恒等矩阵（反对角线全 1）
    效果: 前向仅解 p 个相干源，FBSS 可解 ⌈3p/2⌉ 个 —— 数据量 1/2 换去相干
    上限，导波/通信多径场景默认选它。
    返回: (R_fb, p)，用法同 spatial_smoothing。
    溯源: 2022A（H01）。
    """
    R_f, p = spatial_smoothing(X, subarray_len)
    J = np.fliplr(np.eye(p))
    return (R_f + J @ R_f.conj() @ J) / 2.0, p


def toeplitz_avg_cov(R):
    """Toeplitz 对角平均: 理想 ULA 协方差必为 Toeplitz，退化时对角平均恢复满秩。

    数学核心: 均匀线阵的 R[i,j] 只应依赖 |i−j|；相干/多径退化后做对角平均:
      R̂[i, i+lag] = mean_k R[k, k+lag]（lag = 0, 1, …, M−1），副对角对称共轭。
    注意: 存在阵元幅相误差时 Toeplitz 化会失去误差结构，需先自校准（见 ALS）。
    返回: Toeplitz 化后的协方差（M×M）。
    溯源: 2022A（H01）。
    """
    M = R.shape[0]
    Rout = np.zeros_like(R)
    for lag in range(M):
        v = np.mean([R[i, i + lag] for i in range(M - lag)])
        for i in range(M - lag):
            Rout[i, i + lag] = v
            Rout[i + lag, i] = np.conj(v)
    return Rout


# ============================================================================
# 四、幅相误差自校准（交替最小二乘 ALS）—— 阵元级系统误差的闭环
# ============================================================================

def als_self_calibration(X, n_sources, n_iter=50, tol=1e-6, theta0=None):
    """幅相误差自校准（ALS 骨架）: 阵元增益/相位未知时联合估计信号与误差。

    数学核心（观测模型 X = Γ·A(θ)·S + E，Γ = diag(g) 为未知幅相误差；两半交替）:
      [E 步] 固定 Γ，解信号侧最小二乘: min_{S,θ} ‖X − Γ·A(θ)·S‖²
             （θ 用上一步子空间法/网格粗估 + 局部牛顿步精化）
      [C 步] 固定 A(θ)·S = 信号协方差结构，逐阵元闭式更新 Γ:
             Γ̂ₘ = [X·Sᴴ·Aᴴ]ₘₘ / [A·S·Sᴴ·Aᴴ]ₘₘ  （逐阵元比例，一次矩阵运算）
      循环至 ‖Γ 变化‖ < tol 或 n_iter；Γ(0) = I，即从不带误差的 MUSIC 起步。
    降级说明: 不校准直接 MUSIC 也能交差，论文注明「阵元误差未建模」即可；
    本函数为结构占位，实现时按上述两半填入。
    返回: (θ_est, Γ_diag)——角度估计 + 校准对角阵。
    溯源: 2022A（H01）—— 校准与估计联合问题的标准交替求解。
    """
    raise NotImplementedError(
        "骨架占位：按 docstring 的两半交替实现 —— ①Γ=I、θ0 用 MUSIC/粗网格起步；"
        "②[E 步] 固定 Γ 用阻尼牛顿/扫描更新 θ；③[C 步] 用 Γ̂ₘ 闭式（见上）更新；"
        "④收敛后返回 (θ_est, Γ_diag)。\n"
        "数学内核: Γ̂ₘ = [X·Sᴴ·Aᴴ]ₘₘ / [A·S·Sᴴ·Aᴴ]ₘₘ（逐阵元比例闭式）。")


# ============================================================================
# 五、稀疏重构 CS（OMP 完整实现 / L1 占位）—— 超分辨的第三条路线
# ============================================================================

def omp_reconstruct(A, y, k, tol=1e-6):
    """OMP 贪婪稀疏重构: 从过完备字典中挑 k 个原子线性逼近观测。

    数学核心（三步循环，最多 k 次）:
      选择: i* = argmax_i |⟨r, a_i⟩|（残差与全部原子内积最大者）
      投影: x̂ = argmin ‖y − A_S‖²（对已选原子集合 S 做 LS）
      更新: r ← y − A_S·x̂；‖r‖ < tol 或 |S| = k 时停
    字典原子 = 频点/角度网格上的导向矢量时，x̂ 非零位置即参数估计 ——
    与 MUSIC 高分辨互为对照路线（网格稀疏则分辨率取决于网格密度）。
    参数: A (m×n 字典，n ≫ m 过完备)；y (m 维观测)；k 稀疏度上限。
    返回: (x̂, support)——稀疏系数 + 选中原子下标。
    溯源: 2022A（H01）—— 稀疏重构与子空间法双路线论证。
    """
    m, n = A.shape
    r = y.copy()
    S = []
    x = np.zeros(n, dtype=complex if np.iscomplexobj(A) else float)
    for _ in range(k):
        corr = np.abs(A.conj().T @ r)
        if S:
            corr[S] = -1.0                               # 已选原子不再入选
        i = int(np.argmax(corr))
        S.append(i)
        xS = np.linalg.lstsq(A[:, S], y, rcond=None)[0]
        r = y - A[:, S] @ xS
        if np.linalg.norm(r) < tol:
            break
    x[S] = np.linalg.lstsq(A[:, S], y, rcond=None)[0]
    return x, np.asarray(S, dtype=int)


def l1_sparse_reconstruct(A, y, mu=0.01):
    """L1 正则稀疏重构（LASSO 通道）: min_x ‖y−Ax‖² + μ‖x‖₁。

    数学核心: 凸优化，坐标下降解（sklearn Lasso 默认实现）；复数数据须先按
    实/虚部分裂成两倍维度再解（Lasso 不支持复数）。基追踪（= 无噪声严格
    重构）可改写为 LP: min 1ᵀ(u+v) s.t. A(u−v) ≈ y, u,v ≥ 0。
    与 OMP 并列输出「两种方法一致 → 结论稳健」是 CS 题的标准论证。
    返回: 稀疏系数向量（长度 = 字典原子数）。
    溯源: 2022A（H01）。
    """
    from sklearn.linear_model import Lasso
    Xr = np.vstack([A.real, A.imag]) if np.iscomplexobj(A) else A
    yr = np.concatenate([y.real, y.imag]) if np.iscomplexobj(y) else y
    reg = Lasso(alpha=mu, max_iter=5000).fit(Xr, yr)
    return reg.coef_