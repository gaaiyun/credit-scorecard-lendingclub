# -*- coding: utf-8 -*-
"""评分卡开发的核心函数：分箱、WOE/IV、刻度、KS/PSI/CSI、策略表。

**为什么不用 toad / scorecardpy**
这两个包一行就能出评分卡，但面试官问「你的分箱是怎么切的、单调性怎么保证的、
WOE 公式写一下」，调包答不上来。这里的分箱、WOE、刻度全部手写，每一步都能讲清楚。

Windows 控制台是 GBK，本模块不 print emoji，输出一律 ASCII 标记。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

EPS = 1e-10

# 缺失值统一的箱编号。缺失单独成箱而不是填均值——缺失本身经常就是信号
# （例如 mths_since_last_delinq 为空，意味着这个人从来没逾期过，不是"未知"）。
MISSING_BIN = -1
# 训练集没见过的类别落到这个箱，WOE 取 0（等价于"与整体一致"），比报错稳。
UNSEEN_BIN = -2


# ================================================================ 数值分箱

def tree_bins(x: pd.Series, y: pd.Series, max_bins: int = 6,
              min_rate: float = 0.05) -> list[float]:
    """用单变量决策树找切点。

    为什么用决策树而不是等频：等频分箱不看 y，切出来的箱坏账率可能不单调，
    做出来的评分卡「分数越高越坏」，业务上没法解释。决策树按信息增益切，
    天然更贴目标变量。

    min_rate: 每箱最少占比，防止切出只有几十个人的箱——那种箱的坏账率是噪声。
    返回内部切点（不含 ±inf）。
    """
    ok = x.notna()
    if ok.sum() == 0 or y[ok].nunique() < 2:
        return []
    tree = DecisionTreeClassifier(
        max_leaf_nodes=max_bins,
        min_samples_leaf=max(int(len(x) * min_rate), 50),
        random_state=42,
    )
    tree.fit(x[ok].to_frame(), y[ok])
    thr = tree.tree_.threshold[tree.tree_.feature >= 0]
    return sorted(float(t) for t in thr)


def monotonic_bins(x: pd.Series, y: pd.Series, max_bins: int = 6,
                   min_rate: float = 0.05) -> list[float]:
    """先用决策树切，再逐步合并相邻箱，直到坏账率单调。

    单调性是评分卡对分箱的**质量要求**，不是淘汰变量的理由——
    一个变量不单调，通常是箱切太细导致的噪声，合箱就好了。

    （v1 在这里踩过坑：把"坏账率不单调"当成淘汰变量的标准，26 个变量丢了 25 个，
    最后只剩 1 个变量进模型。正确做法是合箱。）

    合并策略：每轮找出破坏单调方向的相邻对，合掉其中坏账率差最小的那一对
    （差得越小，合并损失的信息越少）。
    """
    cuts = tree_bins(x, y, max_bins=max_bins, min_rate=min_rate)
    while len(cuts) >= 2:
        t = bin_table(apply_bins(x, cuts), y)
        r = t[t["bin"] >= 0].sort_values("bin")["bad_rate"].to_numpy()
        if len(r) < 3:
            break
        d = np.diff(r)
        if (d >= 0).all() or (d <= 0).all():
            break
        # 多数方向即为期望的单调方向，逆着它的那些相邻对是违规对
        up = (d > 0).sum() >= (d < 0).sum()
        bad_idx = np.where(d < 0)[0] if up else np.where(d > 0)[0]
        if len(bad_idx) == 0:
            break
        drop = int(bad_idx[np.argmin(np.abs(d[bad_idx]))])
        cuts.pop(drop)          # 去掉这个切点 = 合并它两侧的箱
    return cuts


def apply_bins(x: pd.Series, cuts: list[float]) -> pd.Series:
    """按切点打箱。缺失单独成一箱（编号 MISSING_BIN = -1）。"""
    edges = [-np.inf] + list(cuts) + [np.inf]
    b = pd.cut(x, bins=edges, labels=False, right=True)
    return pd.Series(b, index=x.index).fillna(MISSING_BIN).astype(int)


# ================================================================ 类别分箱

def cat_bins(x: pd.Series, y: pd.Series, min_share: float = 0.02,
             max_bins: int = 6) -> dict:
    """类别变量分箱：按坏账率排序后合并相邻类别，小类先并。

    返回 {类别值: 箱号}，箱号按坏账率**升序**编号（0 号箱最好）。

    注意一个诚实的说明：类别变量的"单调"是**构造出来的**——我按训练集坏账率排序编号，
    所以它在训练集上必然单调，这没有业务方向上的含义。它是否真的稳定，
    只能靠 OOT 表现和 CSI 来检验。数值变量的单调性才带业务解释
    （比如"额度使用率越高越坏"），两者不能混为一谈。

    min_share: 占比低于此值的类别先合并（小类的坏账率是噪声）。
    """
    d = pd.DataFrame({"x": x.astype("object"), "y": y})
    d = d[d["x"].notna()]
    if len(d) == 0:
        return {}

    g = d.groupby("x")["y"].agg(total="count", bad="sum")
    g["share"] = g["total"] / g["total"].sum()
    g["bad_rate"] = g["bad"] / g["total"]
    g = g.sort_values("bad_rate")

    # 逐步合并：先把占比过小的类别并进坏账率最接近的邻居，再按 max_bins 收口
    groups = [[c] for c in g.index]
    stats = [(g.loc[c, "total"], g.loc[c, "bad"]) for c in g.index]

    def rate(i):
        return stats[i][1] / max(stats[i][0], 1)

    n_total = g["total"].sum()

    def merge(i, j):
        groups[i] = groups[i] + groups[j]
        stats[i] = (stats[i][0] + stats[j][0], stats[i][1] + stats[j][1])
        groups.pop(j)
        stats.pop(j)

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for i in range(len(groups)):
            if stats[i][0] / n_total < min_share:
                # 并进坏账率最接近的相邻组（排序后相邻即坏账率最接近）
                if i == 0:
                    merge(0, 1)
                elif i == len(groups) - 1:
                    merge(i - 1, i)
                else:
                    j = i - 1 if abs(rate(i) - rate(i - 1)) <= abs(rate(i) - rate(i + 1)) else i + 1
                    merge(min(i, j), max(i, j))
                changed = True
                break

    while len(groups) > max_bins:
        diffs = [abs(rate(i + 1) - rate(i)) for i in range(len(groups) - 1)]
        k = int(np.argmin(diffs))
        merge(k, k + 1)

    mapping = {}
    for b, cats in enumerate(groups):
        for c in cats:
            mapping[c] = b
    return mapping


def apply_cat_bins(x: pd.Series, mapping: dict) -> pd.Series:
    """按类别映射打箱。缺失 -> MISSING_BIN，训练集没见过的类别 -> UNSEEN_BIN。"""
    b = x.astype("object").map(mapping)
    b = b.where(x.notna(), MISSING_BIN)             # 缺失
    b = b.where(~(x.notna() & b.isna()), UNSEEN_BIN)  # 见过数据但没见过这个类别
    return b.astype(int)


# ================================================================ WOE / IV

def bin_table(binned: pd.Series, y: pd.Series, name: str = "") -> pd.DataFrame:
    """一个变量的分箱明细表：每箱人数、坏客户数、坏账率、WOE、IV 贡献。

    WOE = ln( (箱内坏/总坏) / (箱内好/总好) )
    正的 WOE 表示这一箱比整体更坏。
    IV = Σ (坏占比 − 好占比) × WOE
    """
    df = pd.DataFrame({"bin": np.asarray(binned), "y": np.asarray(y)})
    g = df.groupby("bin", observed=True)["y"].agg(total="count", bad="sum")
    g["good"] = g["total"] - g["bad"]
    g["bad_rate"] = g["bad"] / g["total"]

    bad_pct = g["bad"] / max(g["bad"].sum(), EPS)
    good_pct = g["good"] / max(g["good"].sum(), EPS)
    # 某一箱全好或全坏时对数会炸，用 0.5 的 Haldane 修正，别直接丢箱。
    bad_pct = bad_pct.replace(0, 0.5 / max(g["bad"].sum(), 1))
    good_pct = good_pct.replace(0, 0.5 / max(g["good"].sum(), 1))

    g["woe"] = np.log(bad_pct / good_pct)
    g["iv_part"] = (bad_pct - good_pct) * g["woe"]
    g["var"] = name
    g["share"] = g["total"] / g["total"].sum()
    return g.reset_index()


def iv_of(tbl: pd.DataFrame) -> float:
    return float(tbl["iv_part"].sum())


def is_monotonic(tbl: pd.DataFrame) -> bool:
    """非缺失箱的坏账率是否单调。"""
    r = tbl[tbl["bin"] >= 0].sort_values("bin")["bad_rate"].to_numpy()
    if len(r) < 3:
        return True
    d = np.diff(r)
    return bool((d >= 0).all() or (d <= 0).all())


def woe_map(tbl: pd.DataFrame) -> dict:
    """分箱明细表 -> {(变量, 箱号): WOE}。"""
    return {(r["var"], int(r["bin"])): float(r["woe"]) for _, r in tbl.iterrows()}


# ================================================================ 评估

def ks_auc(y_true, score) -> tuple[float, float]:
    """KS 与 AUC。score 是「分数越高越好」的评分卡分，
    所以算 AUC / KS 时统一按 -score（即坏客户方向）来算。"""
    from sklearn.metrics import roc_auc_score, roc_curve

    y_true = np.asarray(y_true)
    s = np.asarray(score, dtype=float)
    auc = roc_auc_score(y_true, -s)          # 分越低越坏
    fpr, tpr, _ = roc_curve(y_true, -s)
    ks = float(np.max(tpr - fpr))
    return ks, float(auc)


def ks_auc_prob(y_true, prob_bad) -> tuple[float, float]:
    """输入是坏客户概率时用这个（概率越高越坏）。"""
    return ks_auc(y_true, -np.asarray(prob_bad, dtype=float))


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """群体稳定性指标。切点按 expected 的分位数定，再拿 actual 去套。

    PSI = Σ (实际占比 − 预期占比) × ln(实际占比 / 预期占比)
    经验判读：< 0.1 稳定，0.1–0.25 需关注，> 0.25 说明人群变了、模型要重训。
    """
    return float(psi_detail(expected, actual, bins)["psi_part"].sum())


def psi_detail(expected: np.ndarray, actual: np.ndarray,
               bins: int = 10) -> pd.DataFrame:
    """PSI 的逐箱明细，用来回答"到底哪一段人群变了"。"""
    qs = np.unique(np.percentile(np.asarray(expected, dtype=float),
                                 np.linspace(0, 100, bins + 1)))
    qs[0], qs[-1] = -np.inf, np.inf
    e = pd.cut(pd.Series(expected), qs).value_counts(normalize=True).sort_index()
    a = pd.cut(pd.Series(actual), qs).value_counts(normalize=True).sort_index()
    e = e.replace(0, EPS)
    a = a.replace(0, EPS)
    out = pd.DataFrame({"区间": e.index.astype(str), "预期占比": e.to_numpy(),
                        "实际占比": a.to_numpy()})
    out["psi_part"] = (out["实际占比"] - out["预期占比"]) * \
        np.log(out["实际占比"] / out["预期占比"])
    return out


def csi(expected_bins, actual_bins) -> float:
    """特征稳定性指标：同一个变量，在建模样本和新样本上各箱占比的漂移。

    数学上和 PSI 同一个式子，区别是 PSI 算的是**分数**的分布漂移、
    CSI 算的是**单个变量**的分布漂移。分数 PSI 超标时，看 CSI 才知道
    是哪几个变量在动——这是"人群变了还是口径变了"的第一层排查。
    """
    return float(csi_detail(expected_bins, actual_bins)["csi_part"].sum())


def csi_detail(expected_bins, actual_bins) -> pd.DataFrame:
    e = pd.Series(np.asarray(expected_bins)).value_counts(normalize=True)
    a = pd.Series(np.asarray(actual_bins)).value_counts(normalize=True)
    idx = sorted(set(e.index) | set(a.index))
    e = e.reindex(idx).fillna(0.0).replace(0, EPS)
    a = a.reindex(idx).fillna(0.0).replace(0, EPS)
    out = pd.DataFrame({"箱": idx, "建模期占比": e.to_numpy(),
                        "新样本占比": a.to_numpy()})
    out["csi_part"] = (out["新样本占比"] - out["建模期占比"]) * \
        np.log(out["新样本占比"] / out["建模期占比"])
    return out


# ================================================================ 刻度

def scale_params(pdo: float = 20, base_score: float = 600,
                 base_odds: float = 50) -> tuple[float, float]:
    """把 logit 转成分数的 A、B 两个常数。

    约定：odds = 好/坏。base_odds=50 表示「好:坏 = 50:1 的客户得 base_score=600 分」，
    odds 每翻一倍加 pdo=20 分。

        score      = A - B * ln(odds_bad)        其中 odds_bad = p/(1-p)
        B          = pdo / ln2
        A          = base_score - B * ln(base_odds)

    推导：锚点客户的 好:坏 = base_odds，即 odds_bad = 1/base_odds。
        base_score = A - B * ln(1/base_odds) = A + B * ln(base_odds)
        =>  A = base_score - B * ln(base_odds)

    校验（PDO=20, 600 @ 50:1）：B=28.8539, A=487.12
        好:坏 = 50:1  -> p_bad=1/51  -> score = 600.00
        好:坏 = 100:1 -> p_bad=1/101 -> score = 620.00  （odds 翻倍 +20 分）

    >>> import math
    >>> A, B = scale_params(20, 600, 50)
    >>> round(A, 4), round(B, 4)
    (487.1229, 28.8539)
    >>> round(A - B * math.log((1/51) / (1 - 1/51)), 2)      # 好:坏 = 50:1
    600.0
    >>> round(A - B * math.log((1/101) / (1 - 1/101)), 2)    # odds 翻倍
    620.0

    ---
    这里原先写成 A = base_score + B*ln(base_odds)，锚错了一侧：
    那个公式实际把 **坏:好 = 50:1**（p_bad=98% 的客户）定在 600 分，
    与文档声称的口径正好相反，整条分数虚高 2*B*ln(50) = 225.75 分。
    KS / AUC / PSI / 策略表结构都是分数的单调变换不受影响，
    但「600 分对应 odds 50:1」这句话是假的，面试官一算就穿。已修正。
    """
    B = pdo / np.log(2)
    A = base_score - B * np.log(base_odds)
    return float(A), float(B)


def to_score(prob_bad: np.ndarray, A: float, B: float) -> np.ndarray:
    """坏客户概率 -> 评分卡分数（分越高越好）。"""
    p = np.clip(np.asarray(prob_bad, dtype=float), EPS, 1 - EPS)
    return A - B * np.log(p / (1 - p))


# ================================================================ 建模

def fit_logit(X: pd.DataFrame, y: pd.Series, weights=None, method: str = "IRLS"):
    """逻辑回归。weights 不为空时用频数权重——拒绝推断的重加权法要用到。

    用 GLM(Binomial) 而不是 Logit，是因为 statsmodels 的 Logit 不接受样本权重，
    GLM 的 freq_weights 可以。两者在无权重时结果一致。

    默认用 IRLS（statsmodels 对 GLM 的标准解法，直接给出基于观测信息阵的标准误）。
    IRLS 每轮要对 n x k 的设计矩阵做一次 SVD 求伪逆，内存开销与 n 成正比；
    在提交限额紧的机器上会 MemoryError，此时由调用方降级到 lbfgs
    （纯梯度法，内存是常数级，但收敛精度略松，系数会差到 1e-3 量级）。
    """
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    kw = {} if weights is None else {
        "freq_weights": np.asarray(weights, dtype=float)}
    m = sm.GLM(y, Xc, family=sm.families.Binomial(), **kw)
    if method == "IRLS":
        return m.fit()
    # max_start_irls=0 很关键：statsmodels 的梯度法默认先跑 3 轮 IRLS 热启动，
    # 那 3 轮照样要做 SVD，等于没绕开。置 0 才是纯 lbfgs。
    return m.fit(method=method, maxiter=1000, disp=0, max_start_irls=0,
                 pgtol=1e-9, factr=10.0)


def drop_bad_signs(X: pd.DataFrame, y: pd.Series, weights=None,
                   p_ceil: float = 0.05, min_feats: int = 3,
                   log=print) -> tuple[object, list[str]]:
    """逐步剔除：系数为负（WOE 越大越坏，系数必须为正）或 p 值不显著的变量。

    这一步是评分卡和普通分类模型最大的区别——业务上不接受一个
    「坏客户特征越强、分数反而越高」的变量，哪怕它能提升 AUC。
    """
    import gc

    def _fit(fs):
        try:
            return fit_logit(X[fs], y, weights, method="IRLS")
        except MemoryError:     # 提交限额不够时降级，日志里会留痕
            gc.collect()
            log("    [降级] IRLS 内存不足，本轮改用 lbfgs")
            return fit_logit(X[fs], y, weights, method="lbfgs")

    feats = list(X.columns)
    while True:
        m = _fit(feats)
        bad = [f for f in feats if m.params[f] < 0 or m.pvalues[f] > p_ceil]
        if not bad or len(feats) <= min_feats:
            return m, feats
        worst = max(bad, key=lambda f: m.pvalues[f])
        log(f"    剔除 {worst}（coef={m.params[worst]:+.4f}, p={m.pvalues[worst]:.4f}）")
        feats.remove(worst)
        # 每轮的 GLMResults 都攥着一份 n x k 的设计矩阵副本。不显式释放，
        # 十几轮下来能吃掉 3GB，在提交限额紧的机器上直接 MemoryError。
        del m
        gc.collect()


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """方差膨胀因子。经验阈值 10，超了说明共线严重、系数不可信。"""
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    import statsmodels.api as sm
    Xc = sm.add_constant(X, has_constant="add")
    rows = [{"变量": c,
             "VIF": round(float(variance_inflation_factor(Xc.to_numpy(), i)), 3)}
            for i, c in enumerate(Xc.columns) if c != "const"]
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


# ================================================================ 策略

def cutoff_table(score: np.ndarray, y: np.ndarray, n: int = 10) -> pd.DataFrame:
    """按分数十等分做策略表：每档人数、坏账率，以及「分数≥本档下界」时的
    累计通过率与累计坏账率。选 cutoff 就是在这张表上做通过率与坏账率的权衡。
    """
    df = pd.DataFrame({"score": np.asarray(score, dtype=float),
                       "y": np.asarray(y)})
    df["grp"] = pd.qcut(df["score"], n, labels=False, duplicates="drop")

    g = (df.groupby("grp")
           .agg(下界=("score", "min"), 上界=("score", "max"),
                人数=("y", "count"), 坏客户=("y", "sum"))
           .sort_index(ascending=False))          # 高分档在上
    g["本档坏账率"] = g["坏客户"] / g["人数"]
    g["累计人数"] = g["人数"].cumsum()
    g["累计坏客户"] = g["坏客户"].cumsum()
    g["通过率"] = g["累计人数"] / g["人数"].sum()
    g["累计坏账率"] = g["累计坏客户"] / g["累计人数"]
    return g.reset_index(drop=True)
