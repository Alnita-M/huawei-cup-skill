# -*- coding: utf-8 -*-
"""
image_preprocess.py —— 图像/视频类预处理模板库（华为杯解题 Skill v0.1）
=====================================================================
【适用场景】监控视频前景提取、星图/单幅图像量测、视频能见度估计等图像视觉题的前处理段。
【溯源题号】2017D（基于监控视频的前景目标提取）、2019C（视觉情报信息分析）、
           2020E（能见度估计与预测）、2020C（脑信号，视觉化延伸参考）。
【使用方法】以下均为函数骨架：先按 docstring 的输入输出约定改写补全业务逻辑，
           再嵌入你的流水线。import 均来自 numpy/scipy/sklearn/opencv 常规库，按需裁剪。
【设计原则】① 核心算法（RPCA 的 ADMM 迭代、仿射对齐、灭点/单应估计）给出数学步骤注释，
           直接照抄公式即可落地；② 输入输出一律 numpy 数组。
"""

import numpy as np

# ============================================================================
# 一、基础读取与张量化（2017D 阶段0：数据准备）
# ============================================================================

def video_to_matrix(frames):
    """把 t 帧灰度图堆叠为观测矩阵 D ∈ R^(m×n)，m=w·h, n=t。

    背景低秩 + 前景稀疏的一切矩阵分解都建立在这个"视频=矩阵"的表示上。
    参数:
        frames: np.ndarray (t, h, w)，[0,1] 灰度帧序列
    返回:
        D: np.ndarray (m, n)，每列 = 一帧的按列展开向量
    溯源: 2017D §2 假设1、§3 通用符号。
    """
    t, h, w = frames.shape
    D = frames.reshape(t, h * w).T          # (m, n)
    return D


def matrix_to_frame(D, t, h, w):
    """观测矩阵第 t 列还原为一帧图像（逆操作）。"""
    return D[:, t].reshape(h, w)


def spatial_downsample(frames, factor=2):
    """大分辨率视频先降采样控制 SVD 规模（粗分解→上采样精修两段式）。

    溯源: 2017D §四阶段0 "按需空间降采样"。
    """
    import cv2
    return np.stack([cv2.resize(f, (f.shape[1] // factor, f.shape[0] // factor))
                     for f in frames])


def denoise_frame(img, method="bilateral"):
    """单帧去噪骨架。

    参数:
        img: (h, w) 或 (h, w, c)
        method: 'median' | 'gaussian' | 'bilateral'（边缘保持，监控场景首选）
    溯源: 2017D 噪声假设（零均值小方差）；2019D 滑动平均同族思想（时间域）。
    """
    import cv2
    if method == "median":
        return cv2.medianBlur(img, 5)
    if method == "gaussian":
        return cv2.GaussianBlur(img, (5, 5), 0.8)
    return cv2.bilateralFilter(img, 9, 75, 75)


# ============================================================================
# 二、基线背景建模（2017D 问题1：静态背景，全部为对照/快速通道）
# ============================================================================

def frame_difference_mask(frames, thresh=0.05):
    """帧差法：|I_t − I_{t−1}| > thresh 判为前景（最快基线）。

    局限（论文中要点名）：目标内部产生空洞、静止目标漏检、噪声敏感。
    溯源: 2017D §3 问题1 候选方案对比表。
    """
    diff = np.abs(np.diff(frames, axis=0))
    return (diff > thresh).astype(np.uint8)


def median_background_mask(frames, thresh=0.08):
    """时间中值背景：B = median_t(I_t)，|I_t − B| > thresh 为前景（无参数基线）。

    溯源: 2017D §3 问题1 第二基线。
    """
    B = np.median(frames, axis=0)
    return (np.abs(frames - B) > thresh).astype(np.uint8)


# ============================================================================
# 三、RPCA 系：低秩 + 稀疏分解（2017D 问题1 主线 / 问题2 三分量扩展）
# ============================================================================

def rpca_ialm(D, lam=None, mu0=1e-2, rho=1.5, eps=1e-7, max_iter=300):
    """鲁棒主成分分析（PCP）—— IALM/ADMM 求解主成分追踪。

    模型:  min_{A,S} ‖A‖_* + λ‖S‖_1   s.t.  D = A + S
          ‖A‖_* 核范数 = Σσ_i(A)（背景低秩），‖S‖_1（前景稀疏）。
    λ 经典取 1/sqrt(max(m,n))，实验可在 [0.5, 2]×λ 网格微调。
    迭代步骤（照抄即可）:
      1) A ← SVT(D − S + Y/μ, 1/μ)      # 奇异值收缩：SVD 后奇异值软阈值 (σ−1/μ)_+
      2) S ← shrink(D − A + Y/μ, λ/μ)   # 逐元素软阈值 sign(x)·max(|x|−λ/μ, 0)
      3) Y ← Y + μ(D − A − S)           # 乘子上升
      4) μ ← ρ·μ                        # 罚参数递增
      收敛: ‖D−A−S‖_F / ‖D‖_F < eps
    加速技巧: 小维度 SVD（转置）、随机化/截断 SVD（背景秩很低）。
    参数: D (m, n)（n=帧数）；返回 A(低秩背景), S(稀疏前景)。
    溯源: 2017D §3 问题1（参考文献[4] PCP）。
    """
    m, n = D.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    A = np.zeros_like(D); S = np.zeros_like(D); Y = np.zeros_like(D)
    mu = mu0
    for _ in range(max_iter):
        # 步骤1: 奇异值收缩
        U, s, Vt = np.linalg.svd(D - S + Y / mu, full_matrices=False)
        s_shrunk = np.maximum(s - 1.0 / mu, 0.0)
        A = (U * s_shrunk) @ Vt
        # 步骤2: 稀疏软阈值
        X = D - A + Y / mu
        S = np.sign(X) * np.maximum(np.abs(X) - lam / mu, 0.0)
        # 步骤3-4: 乘子上升与参数更新
        Y = Y + mu * (D - A - S)
        mu *= rho
        if np.linalg.norm(D - A - S, 'fro') <= eps * np.linalg.norm(D, 'fro'):
            break
    return A, S


def rpca_three_component(D, W=None, lam=None, gamma=1.0, mu0=1e-2, rho=1.5,
                         max_iter=300, eps=1e-7):
    """加权三分量分解：低秩背景 A + 稀疏前景 S + 结构化噪声 N（动态背景场景）。

    模型: min ‖A‖_* + λ‖S‖_1 + (γ/2)‖W∘N‖_F²   s.t.  D = A + S + N
    与两分量 PCP 的区别：N 用（可加权）Frobenius 弱惩罚，恰好容纳
    "树叶摇动/水波/喷泉"这类多而小、空间破碎的中幅扰动，避免被 L1 误吞进 S。
    权重 W: 由 temporal_variance_map() 得到，波动大的区域权重小（更容忍噪声）。
    ADMM 三块交替，步骤 3 的 N 更新为逐元素闭式解（加权情形收缩系数各元素不同）:
      N ← (W² / (W² + μγ)) ∘ (D − A − S + Y/μ)      # 来自 ∂/∂N 置零
    参数: W (m, n) 逐像素权重矩阵（None 则全 1）。
    溯源: 2017D §3 问题2（参考文献[5][6] 含未知噪声的鲁棒矩阵分解）。
    """
    m, n = D.shape
    if lam is None:
        lam = 1.0 / np.sqrt(max(m, n))
    if W is None:
        W = np.ones_like(D)
    W2 = W ** 2
    A = np.zeros_like(D); S = np.zeros_like(D); N = np.zeros_like(D)
    Y = np.zeros_like(D); mu = mu0
    for _ in range(max_iter):
        U, s, Vt = np.linalg.svd(D - S - N + Y / mu, full_matrices=False)
        A = (U * np.maximum(s - 1.0 / mu, 0.0)) @ Vt
        X = D - A - N + Y / mu
        S = np.sign(X) * np.maximum(np.abs(X) - lam / mu, 0.0)
        Z = D - A - S + Y / mu
        N = (W2 / (W2 + mu * gamma)) * Z            # 逐元素闭式解
        Y = Y + mu * (D - A - S - N); mu *= rho
        if np.linalg.norm(D - A - S - N, 'fro') <= eps * np.linalg.norm(D, 'fro'):
            break
    return A, S, N


def temporal_variance_map(frames, smooth_sigma=2.0):
    """局部时间方差图 V(i,j)：估计各像素波动强度，用于三分量分解的权重 W。

    W = 1/(V+ε)：波动大的区域（水面/树冠）给较小权重 → 更大的噪声容忍门槛。
    静止区几乎不容忍噪声，残差大者必为前景。
    溯源: 2017D §3 问题2 权重设计。
    """
    from scipy.ndimage import gaussian_filter
    V = np.var(frames, axis=0)
    V = gaussian_filter(V, smooth_sigma)
    return 1.0 / (V + 1e-6)


def binarize_sparse_component(S_frame, method="percentile", q=99.5):
    """稀疏分量 S 单帧自适应二值化 → 前景掩码。

    method='percentile': 阈值 δ 取 |S| 的 q 分位（默认 99.5%）
    method='otsu':       最大类间方差自适应阈值
    溯源: 2017D §3 问题1 后处理（掩码自适应阈值）。
    """
    if method == "percentile":
        delta = np.percentile(np.abs(S_frame), q)
    else:  # otsu（skimage 有现成，此处给等价最小化类内方差骨架）
        from scipy.optimize import minimize_scalar
        def within_var(t):
            fg = S_frame[S_frame > t]; bg = S_frame[S_frame <= t]
            if len(fg) == 0 or len(bg) == 0:
                return 1e18
            w = len(fg) / S_frame.size
            return w * np.var(fg) + (1 - w) * np.var(bg)
        delta = minimize_scalar(within_var, bounds=(0, np.percentile(S_frame, 99)),
                                method='bounded').x
    return (np.abs(S_frame) > delta).astype(np.uint8)


def morphological_cleanup(mask):
    """掩码后处理链：开运算去孤立噪点 → 闭运算填内部空洞 → 连通域剔除小目标。

    溯源: 2017D §3 问题1 后处理链（含连通域面积 < θ_min 剔除）。
    """
    import cv2
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
    keep = np.zeros_like(mask)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= 30:        # θ_min 按分辨率调
            keep[labels == i] = 1
    return keep


# ============================================================================
# 四、几何校正：灭点 / 单应 / 仿射对齐（2017D 问题3、2019C、2020E）
# ============================================================================

def estimate_homography_dlt(src_pts, dst_pts):
    """平面单应 DLT 估计：p ~ H q，H ∈ R^(3×3)（≥4 对点，DLT 线性最小二乘）。

    单应来源: H_g = K[r1 r2 t]（地平面到像面）；或地砖网格/路缘等 4+ 点对直接估计。
    数学骨架: 对每对点构造 2×9 的行 (q'⊗p 形式)，堆叠后 SVD 解最小奇异值向量。
    内部(按需裁):
      A_i = [[q_x, q_y, 1, 0, 0, 0, -q_x*p_x, -q_y*p_x, -p_x],
             [0, 0, 0, q_x, q_y, 1, -q_x*p_y, -q_y*p_y, -p_y]]
      H = V[:, -1].reshape(3, 3)   # 最小二乘解
    溯源: 2019C §3.2 ③、§3.1；2017D §3 问题3 方案B（RANSAC 版见下）。
    """
    from scipy.optimize import least_squares
    # 线性解
    A = []
    for (q, p) in zip(src_pts, dst_pts):
        qx, qy = q; px, py = p
        A.append([qx, qy, 1, 0, 0, 0, -qx * px, -qy * px, -px])
        A.append([0, 0, 0, qx, qy, 1, -qx * py, -qy * py, -py])
    A = np.asarray(A)
    _, _, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)
    # 可选: 非线性精修（重投影残差最小）
    def resid(h):
        Hm = h.reshape(3, 3)
        qh = np.hstack([src_pts, np.ones((len(src_pts), 1))]) @ Hm.T
        qh = qh / qh[:, 2:3]
        return (qh[:, :2] - dst_pts).ravel()
    H = least_squares(resid, H.ravel()).x.reshape(3, 3)
    return H


def ransac_affine_match(frames, ref_idx=0, max_iters=500, inlier_thresh=3.0):
    """特征配准 + RANSAC 稳健仿射估计（方案B：两阶段对齐的配准环节）。

    步骤: 1) SIFT/ORB 检测+描述子匹配（每帧 vs 参考帧）
         2) RANSAC 拟合仿射模型 —— 前景目标上的匹配是外点被剔除
            （前提：背景特征点占多数），天然解决"前景干扰配准"
         3) 返回每帧仿射参数 (2×3) 集合；直接 warp 到参考帧、绝不链式传递
    溯源: 2017D §3 问题3 方案B（工程备选 + 方案A 的初始化）。
    """
    import cv2
    sift = cv2.SIFT_create()
    ref = frames[ref_idx]
    kp_r, des_r = sift.detectAndCompute(ref, None)
    H_list = []
    for f in frames:
        kp_f, des_f = sift.detectAndCompute(f, None)
        if des_f is None or des_r is None:
            H_list.append(np.eye(3)); continue
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des_r, des_f, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]
        if len(good) < 4:
            H_list.append(np.eye(3)); continue
        src = np.float32([kp_r[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst = np.float32([kp_f[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                           ransacReprojThreshold=inlier_thresh)
        H_list.append(np.vstack([M, [0, 0, 1]]))
    return H_list


def align_and_decompose(frames, method="affine"):
    """两阶段对齐 + 分解（对齐除全局几何扰动 → 退化为静态背景 RPCA）。

    warp 后堆叠 D(τ) 送入 rpca_ialm；掩码再逆 warp 回原始坐标。
    注意: 长视频分段处理，每段用中位帧做参考帧，段间重叠帧传递变换。
    溯源: 2017D §3 问题3 方案B 全程。
    """
    import cv2
    H_list = ransac_affine_match(frames)
    h, w = frames[0].shape
    aligned = np.stack([cv2.warpPerspective(f, H_list[i], (w, h))
                        for i, f in enumerate(frames)])
    A, S = rpca_ialm(video_to_matrix(aligned))
    # 掩码逆变换回原坐标
    masks = []
    for i in range(len(frames)):
        m = binarize_sparse_component(S[:, i].reshape(h, w))
        masks.append(cv2.warpPerspective(m, np.linalg.inv(H_list[i]), (w, h)))
    return np.stack(masks), H_list


def estimate_vanishing_point(lines):
    """灭点估计：多组平行线（≥2 条）在像面的交点。

    数学骨架: 每对平行线 l1, l2 交于灭点 v = l1 × l2（齐次线坐标叉积）；
    多对交点做鲁棒平均（中位数 / RANSAC），或最小化到各线距离平方和。
    用途: ① 单应外参构造 r1 ~ K⁻¹v；② 2019C 度量三件套的铅垂灭点 v_z；
         ③ 2020E 车道平行标线→消失点→景深映射的前提。
    溯源: 2019C §3.1 灭点标定；2020E §3.10 几何标定。
    """
    # lines: (N, 3) 齐次线坐标（或由两个端点转为齐次线）
    v_list = []
    for i in range(len(lines)):
        for j in range(i + 1, len(lines)):
            v = np.cross(lines[i], lines[j])
            v = v / v[2] if abs(v[2]) > 1e-9 else None
            if v is not None:
                v_list.append(v)
    V = np.asarray(v_list)
    return np.median(V, axis=0)          # 鲁棒平均作为灭点


def depth_map_from_marking(marking_pixels, known_lengths, H_img):
    """标线像素坐标 → 物理景深映射（能见度估计的几何锚）。

    做法: 用已知国标长度的车道标线（如 6m 实线 + 9m 间隔）建立
    "纵向像素位置 y → 物理距离 d(y)" 的查找表/回归，再配合灭点做透视外推；
    国标长度不确定时，用多条平行标线 + 消失点联合自洽求解（不需绝对长度可得相对景深）。
    溯源: 2020E §3.10（假设 A4 国标标线锚点 + 消失点联合标定）。
    """
    ys = marking_pixels.astype(float)
    lr = np.polyfit(ys, known_lengths, deg=2)     # 二次透视模型
    return np.polyval(lr, marking_pixels)


def optical_flow_dense(frames, step=1):
    """稠密光流（Farneback）骨架：运动场用于目标跟踪/运动特征。

    后接: 2017D 问题6 运动方向直方图/速度场特征；2019C 视频量测的帧间位移。
    溯源: 2017D §3 问题6 特征层"运动类"；2019C 任务2/3。
    """
    import cv2
    flows = []
    for t in range(0, len(frames) - step, step):
        flows.append(cv2.calcOpticalFlowFarneback(
            frames[t], frames[t + step], None, 0.5, 3, 15, 3, 5, 1.2, 0))
    return np.stack(flows)


# ============================================================================
# 五、视频诊断路由（2017D 问题4 第一步 —— 类型自适应）
# ============================================================================

def route_video_type(frames):
    """按像素统计自动路由：稳定机位 / 抖动视频 / 动态背景。

    指标: ① 帧间全局配准残差（小→稳定机位；大但仿射可解释→抖动）
         ② 空白时段像素时间方差（大→动态背景）
         ③ 亮度漂移曲线（判断光照类型）
    返回: 'static' | 'jitter' | 'dynamic'，据此选择问题1/3/2 管线。
    溯源: 2017D §四 阶段1 视频诊断路由。
    """
    variance = temporal_variance_map(frames)
    blank_var = np.median(variance[frames.mean(axis=0) < 0.3]) if (frames.mean(axis=0) < 0.3).any() else np.median(variance)
    if blank_var > 0.02:
        return 'dynamic'
    # 帧差均值作为抖动代理（工程简化，正式版用 SIFT 配准残差）
    diff = np.abs(np.diff(frames, axis=0)).mean()
    return 'jitter' if diff > 0.01 else 'static'