# -*- coding: utf-8 -*-
"""拒绝推断：把被拒客户拉回建模样本，纠正幸存者偏差。

**问题是什么**
A 卡只能在已核准客户身上建模，但上线后要对**所有**申请人打分。
已核准样本是被上一版政策筛过的，缺了低分段那一截人群，
模型学到的是"通过人群内部的规律"，外推到全体申请人时会失真。

**两种主流做法**
1. 重加权（Augmentation / Reweighting）
   先建一个「是否被核准」的模型 P(accept|x)，再给每个已核准客户
   一个 1/P(accept|x) 的权重——长得像被拒客户的那些人被放大，
   加权后的样本近似代表全体申请人。用加权样本重训 KGB 得到 AGB。
   前提是 MAR：核准决策只依赖我们观察得到的 x。这个前提通常不成立，
   所以重加权只能纠正「可观察部分」的偏差，报告里必须写明。

2. 打包法（Parceling）
   用只含共同变量的评分卡给被拒客户打分，按分数分组，
   在每组里按 组内已核准客户坏账率 x 倍数 k 的比例随机指定坏客户，
   然后把推断后的被拒样本并进训练集重训。
   k 是个假设（业内常用 2~4），必须单列并做敏感性分析。

**一个必须说清楚的根本局限**
被拒客户的真实表现**永远观察不到**，所以拒绝推断的效果无法被真正验证。
任何「做了拒绝推断后 KS 提升 X」的说法，那个 X 都是在已核准样本上算的，
而已核准样本恰恰是不需要纠偏的那部分。能做的只有：看系数怎么变、
看分数分布怎么变、看同一通过率下换进换出的是哪批人（swap set）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

EPS = 1e-10


def accept_weights(p_accept: np.ndarray, trim_pct: float = 99.0,
                   floor: float = 1e-3) -> np.ndarray:
    """由核准概率算重加权权重 w = 1 / P(accept|x)。

    trim_pct: 权重在此分位数上截尾。核准率只有 5% 量级时，
    极端权重能到几百，一个样本就能主导整条回归线，必须截。
    """
    p = np.clip(np.asarray(p_accept, dtype=float), floor, 1.0)
    w = 1.0 / p
    cap = np.percentile(w, trim_pct)
    return np.minimum(w, cap)


def parcel_bad_flags(reject_score: np.ndarray, accept_score: np.ndarray,
                     accept_y: np.ndarray, k: float = 3.0, n_bands: int = 10,
                     random_state: int = 42) -> np.ndarray:
    """打包法：给被拒客户指定推断坏标签。

    做法：按已核准样本的分数分位切 n_bands 个档，
    每档的推断坏账率 = 该档已核准客户的坏账率 x k（封顶 1.0），
    在该档的被拒客户里按这个比例随机抽出坏客户。

    k 的含义：同一个分数档里，被拒客户比已核准客户坏多少倍。
    它是假设，不是估计——因为真实值永远观察不到。
    """
    rng = np.random.default_rng(random_state)
    edges = np.unique(np.percentile(accept_score, np.linspace(0, 100, n_bands + 1)))
    edges[0], edges[-1] = -np.inf, np.inf

    a_band = pd.cut(accept_score, edges, labels=False)
    r_band = pd.cut(reject_score, edges, labels=False)
    rate = pd.Series(accept_y).groupby(a_band).mean()

    out = np.zeros(len(reject_score), dtype=int)
    for b, r in rate.items():
        idx = np.where(r_band == b)[0]
        if len(idx) == 0:
            continue
        p = min(float(r) * k, 1.0)
        out[idx] = (rng.random(len(idx)) < p).astype(int)
    # 落在已核准分数范围之外的被拒客户（分数更低），按最低档的推断坏账率处理
    lost = np.where(pd.isna(r_band))[0]
    if len(lost):
        p = min(float(rate.min()) * k, 1.0) if len(rate) else 0.0
        out[lost] = (rng.random(len(lost)) < p).astype(int)
    return out


def swap_set(score_a: np.ndarray, score_b: np.ndarray, y: np.ndarray,
             approval_rate: float) -> pd.DataFrame:
    """同一通过率下，两套评分卡的换入换出分析。

    这是拒绝推断为数不多能被**真正验证**的地方：两套卡在同一审批量下
    批的是不是同一批人，换进来的那批坏账率比换出去的高还是低。
    """
    n = len(y)
    k = int(round(n * approval_rate))
    ta = np.sort(score_a)[::-1][k - 1]
    tb = np.sort(score_b)[::-1][k - 1]
    pa, pb = score_a >= ta, score_b >= tb

    both = pa & pb
    only_a = pa & ~pb          # A 批、B 拒：B 换出去的
    only_b = ~pa & pb          # B 批、A 拒：B 换进来的
    rows = []
    for nm, m in (("两套都批", both), ("仅A批(B换出)", only_a),
                  ("仅B批(B换入)", only_b), ("两套都拒", ~pa & ~pb)):
        rows.append({"人群": nm, "人数": int(m.sum()),
                     "占比": round(float(m.mean()), 4),
                     "坏账率": round(float(y[m].mean()), 4) if m.sum() else np.nan})
    return pd.DataFrame(rows)
