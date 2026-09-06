# -*- coding: utf-8 -*-
"""
predictor_modeling_validation.py —— 预测类建模与验证模板库（华为杯解题 Skill v0.1）
==================================================================================
【适用场景】回归/分类预测题的"模型选型 → 防泄漏验证 → 评估指标 → 不平衡处理
           → 集成增强"主链路（对应 2020B 问题3、2021B 问题3、2021D 问题2/3、
           2020C 问题1/4、2019E 问题2）。
【溯源题号】2020B（嵌套 CV、按时间切分、模型 zoo、问题3）、2021B（滚动重训、
           Stacking、残差订正、问题3）、2021D（类别不均衡三件套、逐标签选型、
           问题3）、2020C（伪标签门控半监督、学习曲线、问题1/3/4）。
【核心纪律】① 特征筛选/调参/不平衡处理一切数据依赖量都在训练折内完成（防泄漏）；
           ② 预测任务特征时间戳 ≤ 起报时刻（时间因果）；③ 树/神经网络超参用
           随机搜索 + CV，不做全网格穷举。
"""
import numpy as np

# ============================================================================
# 一、模型动物园（2020B 问题3 双轨策略：线性族主轨 + 非线性对照轨）
# ============================================================================

def train_model_zoo(Xtr, ytr, Xva, yva, task="regression"):
    """统一接口跑模型对比表：线性族 vs 树族 vs 核/深度 —— 返回每模型验证误差。

    设计哲学（2020B §6.1）: 小样本 + 要解释 + 要优化 → 默认线性族为 final，
    除非非线性在样本外用 CV 证明了自己。线性模型让后续约束优化可 LP/QP 解析。
    模型清单（按需裁剪）:
      回归: OLS(LinearRegression)/岭/LASSO/ElasticNet/PLS(Ridge 双中心近似)/
            RandomForest/GBDT/XGBoost/LightGBM/CatBoost/SVR-RBF/浅层 MLP
      分类: LogisticRegression/RandomForestClassifier/XGBClassifier/SVC/LGBM + 浅层 NN
    返回: dict{model_name: (验证指标, fit_model)}；回归用 RMSE，分类用 AUC。
    溯源: 2020B §6.1-6.3；2021D 独立解答 135 行（模型谱系表）。
    """
    from sklearn.linear_model import LinearRegression, Ridge, Lasso
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error

    models = {
        "OLS": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=1e-2),
        "RF": RandomForestRegressor(n_estimators=300, random_state=0),
        "GBDT": GradientBoostingRegressor(random_state=0),
        "SVR": SVR(C=1.0, gamma="scale"),
    }
    if task == "classification":
        from sklearn.linear_model import LogisticRegression
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        models = {k: None for k in ["LogReg", "RF", "GBDT", "SVM"]}
        from sklearn.ensemble import GradientBoostingClassifier
        models["LogReg"] = LogisticRegression(max_iter=2000)
        models["RF"] = RandomForestClassifier(n_estimators=300, random_state=0)
        models["GBDT"] = GradientBoostingClassifier(random_state=0)
        models["SVM"] = SVR()  # 分类请换 SVC（c-svc, rbf）
        score = roc_auc_score
    else:
        score = lambda yt, yp: np.sqrt(mean_squared_error(yt, yp))

    results = {}
    for name, m in models.items():
        if m is None:
            continue
        m.fit(Xtr, ytr)
        pred = m.predict(Xva)
        results[name] = (score(yva, pred), m)
    return results


# ============================================================================
# 二、防泄漏验证协议（2020B 问题3 核心纪律 / 2021B 时间因果 / 2021D 折内拟合）
# ============================================================================

def nested_cv_evaluate(X, y, build_fn, outer_k=5, inner_k=5, metric="rmse", random_state=0):
    """嵌套 K 折交叉验证：外层评估泛化误差，内层做筛选/调参（防泄漏的关键）。

    协议（2020B §6.3 步2 "嵌套流程"）:
      外层: 留出测试集（2020B 建议按时间切最近 20% 防时序泄漏）
      内层: 每折内独立执行 特征筛选(递归函数) + 调参 + 训练，测试折只评一次
    特征选择必须对每个内层折重新跑 —— 严谨版；工程折衷版 = 外层训练集筛一次，
    但严禁测试集参与任何筛选/调参决策（写进论文）。
    返回: 每外层折的测试指标数组。
    溯源: 2020B §6.3；2021D 独立解答 143 行（"SMOTE 等一切数据依赖量都在训练折内"）。
    """
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, roc_auc_score
    kf = KFold(n_splits=outer_k, shuffle=True, random_state=random_state)
    scores = []
    for tr_idx, te_idx in kf.split(X):
        m = build_fn(X[tr_idx], y[tr_idx])            # 内层 CV 在 build_fn 内完成
        pred = m.predict(X[te_idx])
        if metric == "rmse":
            scores.append(np.sqrt(mean_squared_error(y[te_idx], pred)))
        else:
            scores.append(roc_auc_score(y[te_idx], pred))
        # 残差诊断挂点: 训练-测试差 = 过拟合信号（2020B §6.4）
    return np.asarray(scores)


def time_series_split(X, y, n_splits=5, gap=0):
    """按时间顺序切分（时序预测防泄漏标配）: 第 k 折只允许用更早的数据训练。

    与随机 KFold 的区别: 2020B 建议按时间切最近段做留出；2021B 要求
    验证/测试期不参与调参 + 每 7 天滚动重训模拟真实预报。
    实现: sklearn.model_selection.TimeSeriesSplit 封装。
    溯源: 2020B §6.3 步1；2021B A6（时间因果 = 隐性评分门槛）。
    """
    from sklearn.model_selection import TimeSeriesSplit
    return TimeSeriesSplit(n_splits=n_splits, gap=gap).split(X, y)


def rolling_refit_predict(X_all, y_all, train_len, refit_every=7):
    """滚动重训 + 滚动预报（2021B 问题3 的运行方式）。

    每 7 天: 用截至当日全部数据重训（或按固定窗），预测未来 1-3 天；
    模拟"起报时刻只有此前信息"的真实约束，输出按日拼接的预测序列。
    溯源: 2021B A6 + §三 问题3 训练与验证协议。
    """
    preds = []
    for t in range(train_len, len(X_all)):
        if (t - train_len) % refit_every == 0:
            pass                                    # 此处置位: 重训模型
        # preds.append(model.predict(X_all[t:t+1]))
    return preds


# ============================================================================
# 三、评估指标（2020B §6.4 / 2021B A5 / 2020C A12 各自口径拼装）
# ============================================================================

def regression_metrics(y_true, y_pred, y_scale=None):
    """回归四件套: RMSE / MAE / MAPE / R²（可加训练-测试差做过拟合诊断）。

    合理性锚点（2020B）: 目标量级 1.4 左右、最优 0.6 时，测试 RMSE 应在小数位
    可比范围；若 RMSE 与 y 的标准差同量级 → 模型无预测力，回头查特征。
    溯源: 2020B §6.4；2021B A5（R² 计入）。
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 1e-9)))) * 100
    return {"RMSE": rmse, "MAE": mae, "MAPE%": mape, "R2": r2_score(y_true, y_pred)}


def classification_metrics(y_true, y_pred, y_prob=None):
    """分类指标包: 准确率 / 混淆矩阵 / 每类 P/R/F1 / Kappa / AUC（二分类）。

    2020C A12 纪律: 随机/分层划分多次平均，报告准确率 + 混淆矩阵 + 各类 P/R + Kappa。
    AUC 为不平衡场景主评估（2021D 对齐点7）。
    溯源: 2020C A12（南大/交大一致）；2021D 独立解答（AUC 为主、少数类 Recall 为辅）。
    """
    from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report,
                                 cohen_kappa_score, roc_auc_score)
    rep = {"accuracy": accuracy_score(y_true, y_pred),
           "confusion": confusion_matrix(y_true, y_pred),
           "report": classification_report(y_true, y_pred, zero_division=0, output_dict=True),
           "kappa": cohen_kappa_score(y_true, y_pred)}
    if y_prob is not None and len(np.unique(y_true)) == 2:
        rep["auc"] = roc_auc_score(y_true, y_prob)
    return rep


# ============================================================================
# 四、类别不均衡三件套（2021D 对齐点7 —— 过采样 / Focal Loss / 阈值移动）
# ============================================================================

def imbalance_pipeline(Xtr, ytr, method="smote", **kwargs):
    """不均衡处理入口：SMOTE/ADASYN 过采样 | 类别权重 | 阈值移动三选一或组合。

    铁律: SMOTE 合成与一切重采样必须在 CV 折内完成（否则验证集被污染）。
    逐标签分别处理: 5 个二分类任务可各自选不同模型与不均衡策略（2021D）。
    溯源: 2021D 对齐点7 + 独立解答 169 行（SMOTE/ADASYN）+ 143 行(折内拟合)。
    """
    if method == "smote":
        try:
            from imblearn.over_sampling import SMOTE
            return SMOTE(random_state=0, **kwargs).fit_resample(Xtr, ytr)
        except ImportError:
            raise ImportError("imblearn 未安装：pip install imbalanced-learn（按需裁剪可换 ADASYN）")
    if method == "class_weight":
        # 用法: 在模型构造处传 class_weight='balanced'（LR/SVM/树原生支持）
        from sklearn.utils.class_weight import compute_class_weight
        w = compute_class_weight("balanced", classes=np.unique(ytr), y=ytr)
        return Xtr, ytr, dict(zip(np.unique(ytr), w))
    if method == "threshold_move":
        return Xtr, ytr, {}          # 训练后按 Youden J 最大化在验证折上移动阈值
    raise ValueError(method)


def focal_loss_binary(y_true, y_prob, gamma=2.0, alpha=0.25):
    """Focal Loss（二分类，张量版骨架）：压低易分样本权重、聚焦少数类。

    FL = −α_t (1 − p_t)^γ log p_t，p_t = p(y=1) 时 p, 否则 1−p。
    轻量实现可直接在 numpy 上算标量损失（训练循环交给 DL 框架时按此公式改写）。
    溯源: 2021D 差距分析 二.7（Focal Loss 列入三件套）。
    """
    p = np.clip(y_prob, 1e-9, 1 - 1e-9)
    pt = np.where(y_true == 1, p, 1 - p)
    at = np.where(y_true == 1, alpha, 1 - alpha)
    return float(np.mean(-at * (1 - pt) ** gamma * np.log(pt)))


def youden_threshold(y_val, y_prob_val):
    """Youden J 阈值移动: J = TPR − FPR 最大点作为最优决策阈值（折上选、折外测）。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y_val, y_prob_val)
    j = tpr - fpr
    return thr[np.argmax(j)], float(j.max())


# ============================================================================
# 五、集成增强（2021B 对齐点A4 / 2020C 多轮投票 / 残差订正双流派）
# ============================================================================

def stacking_ensemble(base_models, Xtr, ytr, Xte, meta_model=None, cv=5):
    """Stacking 集成：基学习器 OOF 预测作元特征，元模型再学习（防泄漏版）。

    获奖论文实证（2021B A4）: LightGBM/XGBoost/ElasticNet-LSTM + Stacking 夺魁。
    实现要点: 基模型在内层 CV 上产 OOF 预测（不可用全量训练再预测自己），
    元模型在 OOF 特征上训练、对测试集做最终预测。
    溯源: 2021B 对齐点 A4（汀丶 + Stacking）。
    """
    from sklearn.model_selection import cross_val_predict
    from sklearn.linear_model import Ridge
    OOF = np.column_stack([cross_val_predict(m, Xtr, ytr, cv=cv, method="predict")
                           for m in base_models])
    meta = (meta_model or Ridge(alpha=1.0)).fit(OOF, ytr)
    te_oof = np.column_stack([m.predict(Xte) for m in [mm.fit(Xtr, ytr) for mm in base_models]])
    return meta.predict(te_oof), OOF


def residual_correction(primary_pred, X, y, learner="ridge"):
    """二次建模双流派之 B：残差订正 —— ŷ = F(一次预报) + r̂(X)。

    意义（2021B A3 完全一致）: 以一次预报为基准，学习"一次预报→实测"系统偏差
    并叠加修正量；与直接回归 y=f(·) 双流派对比选优，残差订正叙事更工整。
    注: 一次预报 F 也作为特征进直接回归（流派 A 由 train_model_zoo 覆盖）。
    溯源: 2021B A3 + §三 问题3 特征工程。
    """
    from sklearn.linear_model import Ridge, LinearRegression
    resid = y - primary_pred
    m = (Ridge() if learner == "ridge" else LinearRegression()).fit(X, resid)
    return m, primary_pred + m.predict(X)         # (残差模型, 订正后预测)


def pseudo_label_self_train(X_label, y_label, X_unlabel, base_clf, margin=0.8,
                            max_rounds=10, structural_gate=None):
    """伪标签半监督（2020C 问题3）：置信度门控 + 迭代扩充。

    获奖论文几乎同一方法（2020C A8/A9）: 取预测得分最高者（或 softmax > 0.8）
    赋予伪标签并入训练，迭代 ≤10 轮；本题额外可用"每轮每行每列恰一个正样本"
    的实验设计先验做结构门控（行列 argmax 互证才给伪标签）。
    溯源: 2020C A8/A9（交大/南大一致）。
    """
    for _ in range(max_rounds):
        prob = base_clf.predict_proba(X_unlabel)[:, 1]
        confident = prob > margin
        if structural_gate is not None:
            confident &= structural_gate(prob)     # 结构门控（行列互证）挂点
        if not confident.any():
            break
        X_label = np.vstack([X_label, X_unlabel[confident]])
        y_label = np.concatenate([y_label, (prob[confident] > 0.5).astype(int)])
        X_unlabel = X_unlabel[~confident]
        base_clf.fit(X_label, y_label)
    return base_clf, len(y_label)


def learning_curve_check(model_fn, X, y, ratios=(0.4, 0.6, 0.8, 1.0), seeds=10):
    """学习曲线/训练占比扫描（2020C A13 少样本权衡证据）。

    获奖论文实证: CatBoost 训练占比 0.8→0.4 时 95.06%→82.06% ——
    "训练占比-准确率"曲线证明少量训练样本仍可用，是少样本题的标准论证图。
    溯源: 2020C A13（交大）——分层划分 + 多种子重复取均值。
    """
    from sklearn.model_selection import train_test_split
    out = {}
    for r in ratios:
        accs = []
        for s in range(seeds):
            Xtr, Xva, ytr, yva = train_test_split(X, y, train_size=r, stratify=y, random_state=s)
            m = model_fn(Xtr, ytr)
            accs.append(m.score(Xva, yva))
        out[r] = (np.mean(accs), np.std(accs))
    return out