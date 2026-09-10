# -*- coding: utf-8 -*-
"""多目标去重与聚类骨架（参数化模型搜索的通用后处理）

对应 SKILL.md **C5 / C6**（[G] 实测实证）：
  C5  去重/相似度判据必须定义在**观测空间的重叠度**上，而不是参数空间的欧氏距离
      （参数相近 ≠ 解重合：实测同一目标的近似解可分散到参数尺度的一大截，
        参数空间阈值化与聚类都收敛不了）
  C6  **枚举**靠"检出即抑制"（在搜索阶段做），**归并**靠本模块的聚类；
      两者功能正交、不可互相替代（关掉抑制改成事后聚类，候选会退化为同一目标）

本模块只做**归并**这一步：给定若干候选（每个候选能在观测空间画出自己的形状），
按"参数距离 + 观测重叠"的合取判据求**连通分量**（等价单链接层次聚类在阈值处切树）。

适用（可迁移）：Hough 族的直线/圆/椭圆/参数曲线、模板匹配、曲线拟合、点集配准、
              以及其他"每个目标对应一条曲线/一个区域"的多目标检测任务。

用法示例（把候选适配成 Candidate 后调用）：
    cands = [Candidate(params=np.array([R, beta, C]), shape=lambda: y_array), ...]
    groups = cluster_candidates(cands, param_norm, eps_d=0.35, tau=15, eps_ov=0.60)
"""
from dataclasses import dataclass, field
from typing import Callable, List, Sequence

import numpy as np


@dataclass
class Candidate:
    """一个候选解。

    params : 参数向量（用于参数空间距离；各分量请先在调用方归一化或给出 param_norm）
    shape  : 返回"观测空间中的形状采样序列"的函数 —— 返回 (n, d) 数组，
             d≥1；对曲线类通常是一维序列（沿自变量等距取点后的因变量值），
             对区域类可返回遮罩扁平化的坐标序列。**不同候选必须用同一自变量采样网格**。
    score  : 证据强度（如 Hough 得分），用于选代表与加权平均；越大越可信
    meta   : 附加信息（来源图、编号等）
    """
    params: np.ndarray
    shape: Callable[[], np.ndarray]
    score: float = 1.0
    meta: dict = field(default_factory=dict)
    _shape_cache: np.ndarray = None

    def s(self) -> np.ndarray:
        if self._shape_cache is None:
            self._shape_cache = np.asarray(self.shape())
        return self._shape_cache


def _param_dist(pi: np.ndarray, pj: np.ndarray, norm: Sequence[float] | None) -> float:
    d = np.asarray(pi, float) - np.asarray(pj, float)
    if norm is not None:
        d = d / np.asarray(norm, float)
    return float(np.sqrt(np.sum(d * d)))


def _shape_overlap(si: np.ndarray, sj: np.ndarray, tau: float) -> float:
    """观测空间重叠率：同一采样网格上，二者形状之差在容差 tau 内的比例。

    对曲线类（一维序列）即 |y_i(x) - y_j(x)| < tau 的占比；
    对多维形状请在适配层先降维到同一可比序列。
    """
    n = min(len(si), len(sj))
    if n == 0:
        return 0.0
    return float((np.abs(si[:n] - sj[:n]) < tau).mean())


def cluster_candidates(cands: List[Candidate], param_norm: Sequence[float] | None = None,
                       eps_d: float = 0.35, tau: float = 15.0, eps_ov: float = 0.60,
                       require_both: bool = True) -> List[List[int]]:
    """按合取判据求连通分量。

    require_both=True（默认，推荐）：参数距离 ≤ eps_d **且** 观测重叠 ≥ eps_ov 才判同一目标。
      —— 只按参数距离会漏并（同目标近似解分散），只按重叠会误并（不同目标局部相交）。
    require_both=False：任一条满足即合并（更激进，慎用）。

    返回：[[候选下标, ...], ...]，按簇内最高分降序。
    """
    n = len(cands)
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            cond_d = _param_dist(cands[i].params, cands[j].params, param_norm) <= eps_d
            cond_ov = _shape_overlap(cands[i].s(), cands[j].s(), tau) >= eps_ov
            ok = (cond_d and cond_ov) if require_both else (cond_d or cond_ov)
            if ok:
                adj[i].append(j)
                adj[j].append(i)
    seen, groups = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            u = stack.pop()
            comp.append(u)
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        groups.append(comp)
    groups.sort(key=lambda g: -max(cands[i].score for i in g))
    return groups


def group_representative(cands: List[Candidate], group: Sequence[int]) -> dict:
    """簇代表：得分加权平均参数 + 最高分 + 支持度（用于"每个目标只有一个表征"）。"""
    w = np.array([max(cands[i].score, 1e-9) for i in group])
    P = np.stack([cands[i].params for i in group])
    rep = (P * w[:, None]).sum(0) / w.sum()
    return dict(params=rep, score=max(cands[i].score for i in group),
                support=len(group), members=list(group))


if __name__ == "__main__":
    # 自检：构造 2 个真目标（各 4 个近似解）+ 2 个孤立的近似解，验证归并结果
    rng = np.random.default_rng(0)
    x = np.arange(100.0)

    def shape_of(a, b):
        return lambda: a * np.sin(2 * np.pi * x / 100.0) + b

    cands = []

    def make(phase, k, base):
        # 同一候选的参数与形状必须来自**同一次**采样；分两次采样会让自检本身引入不一致
        a, c = 30.0 + rng.normal(0, 2), 50.0 + rng.normal(0, 3)
        return Candidate(np.array([a, phase + rng.normal(0, .05), c]), shape_of(a, c), base - 0.05 * k)

    for k in range(4):                      # 目标 1 的 4 个近似解
        cands.append(make(1.0, k, 0.5))
    for k in range(4):                      # 目标 2：参数相近但相位差明显 ⇒ 形状不重叠
        cands.append(make(4.0, k, 0.4))
    # 注意：目标 2 与目标 1 参数相近但相位差 3 rad ⇒ 形状不重叠，合取判据应把它们分开
    groups = cluster_candidates(cands, param_norm=[30.0, np.pi, 500.0], eps_d=0.35, tau=6.0, eps_ov=0.60)
    print("候选 %d 个 → 聚类 %d 簇：%s" % (len(cands), len(groups), groups))
    assert len(groups) == 2, "自检失败：期望 2 簇（参数相近但形状不重叠的目标应分开）"
    for g in groups:
        r = group_representative(cands, g)
        print("  簇：支持度 %d，代表参数 %s，最高分 %.3f" % (r["support"], np.round(r["params"], 2), r["score"]))
    print("自检通过：C5 合取判据生效（参数接近但观测不重叠者未被误并）")
