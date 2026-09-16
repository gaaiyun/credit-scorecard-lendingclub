# -*- coding: utf-8 -*-
"""Vintage 账龄分析与好坏客户定义。

**这份数据的一个真实限制，以及绕过它的办法**
Lending Club 只给贷款的**终态快照**（数据拉取时点的 loan_status），不给逐月表现记录。
所以无法直接数「第 N 个月末有多少户 M3+」。

但可以重建：`last_pymnt_d` 是这笔贷款最后一次还款的月份。一个账户在账龄 m 时的
逾期月数 = m − 最后还款账龄（已结清的除外）。据此可以还原任意账龄上的
M0/M1/M2/M3+ 状态，做出真正的 vintage 曲线。

这个重建有三处近似，README 和图注都要写明：
1. 数据只到月粒度，所以逾期是「月」不是「天」，M3+ 对应 90~120 DPD 区间。
2. 提前结清的账户在结清后没有还款记录，必须靠 loan_status 把它们区分出来，
   否则会被误判成「停止还款」。
3. **看不到「还款—逾期—还上」的中途反复**。`last_pymnt_d` 只记最后一次还款，
   一个客户如果逾期 3 个月后又还上了，这里会判成一直正常。所以本项目的口径
   严格说是「终态 M3+」（停了就再没还过），不是银行常用的「曾经 M3+」，
   会略微低估坏客户数。无担保消费贷从 90+ 逾期治愈的比例很低，影响有限，
   但这是口径差异，面试被问到要说清楚，不能含糊成「就是 M3+」。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# 终态：已结清（好）与已核销（坏）
PAID_STATUS = {"Fully Paid", "Does not meet the credit policy. Status:Fully Paid"}
BAD_STATUS = {"Charged Off", "Default",
              "Does not meet the credit policy. Status:Charged Off"}
# 快照时点仍在途的状态
INFLIGHT_STATUS = {"Current", "In Grace Period",
                   "Late (16-30 days)", "Late (31-120 days)"}


def months_between(a: pd.Series, b: pd.Series) -> pd.Series:
    """b − a 的整月数。"""
    return ((b.dt.year - a.dt.year) * 12 + (b.dt.month - a.dt.month)).astype("Float64")


def add_mob(df: pd.DataFrame) -> pd.DataFrame:
    """补上账龄相关字段：
    mob_last_pymnt  最后一次还款发生在第几个账龄月（放款当月 = 0）
    mob_observable  这笔贷款最多能观察到第几个账龄月（受数据快照时点限制）
    is_paid / is_bad_ever  终态标记
    """
    d = df.copy()
    d["mob_last_pymnt"] = months_between(d["issue_dt"], d["last_pymnt_dt"])
    # 一次都没还过（首期即违约）：记为 0，后面按"从第 3 个账龄月起就是 M3+"处理
    d["never_paid"] = d["last_pymnt_dt"].isna()
    d.loc[d["never_paid"], "mob_last_pymnt"] = 0.0
    d["mob_last_pymnt"] = d["mob_last_pymnt"].astype(float).clip(lower=0)

    d["is_paid"] = d["loan_status"].isin(PAID_STATUS)
    d["is_bad_ever"] = d["loan_status"].isin(BAD_STATUS)
    return d


def delinq_months_at(d: pd.DataFrame, mob: int) -> pd.Series:
    """账龄 mob 时点的「连续未还款月数」。

    规则
    ----
    - 在 mob 之前（含）已结清的账户 -> 0（好，不存在逾期）
    - 其余账户 -> max(0, mob − 最后还款账龄)

    注意：已核销账户在核销后同样不再有还款记录，所以它的
    「未还款月数」会随 mob 一直增长，这正是我们想要的。
    """
    paid_off_by_mob = d["is_paid"] & (d["mob_last_pymnt"] <= mob)
    dq = (mob - d["mob_last_pymnt"]).clip(lower=0)
    return dq.where(~paid_off_by_mob, 0.0)


def label_at(d: pd.DataFrame, mob: int, months_delinq: int = 3) -> pd.Series:
    """账龄 mob 上的好/坏/灰标签。

    坏 = M{months_delinq}+ ：连续 months_delinq 个月未还款
    灰 = M1 ~ M{months_delinq-1}：轻度逾期，既不能算坏也不能算好，剔除
    好 = 当期正常或已结清

    返回 'bad' / 'grey' / 'good'
    """
    dq = delinq_months_at(d, mob)
    out = pd.Series("good", index=d.index, dtype="object")
    out[(dq >= 1) & (dq < months_delinq)] = "grey"
    out[dq >= months_delinq] = "bad"
    return out


def observable_mob(d: pd.DataFrame, snapshot: str) -> pd.Series:
    """每笔贷款最多能观察到的账龄月数 = 数据快照月 − 放款月。

    超过这个账龄的表现是看不到的，拿来算坏账率就是右截断的假低值。
    """
    snap = pd.Timestamp(snapshot)
    return ((snap.year - d["issue_dt"].dt.year) * 12 +
            (snap.month - d["issue_dt"].dt.month)).astype(float)


def vintage_table(d: pd.DataFrame, cohort_col: str, max_mob: int = 36,
                  months_delinq: int = 3, snapshot: str = "2019-03") -> pd.DataFrame:
    """cohort × MOB 的累计坏账率表。

    每个 cohort 只算到它**能观察到**的账龄为止，看不到的填 NaN——
    这是 vintage 分析最容易出错的地方：把未成熟 cohort 的曲线画到底，
    会画出一条假的「越晚的人群越好」。
    """
    obs = observable_mob(d, snapshot)
    rows = []
    for c, idx in d.groupby(cohort_col).groups.items():
        g = d.loc[idx]
        g_obs = obs.loc[idx]
        n = len(g)
        for m in range(1, max_mob + 1):
            if (g_obs < m).any():          # 该 cohort 还没走到这个账龄
                rows.append({"cohort": c, "mob": m, "n": n, "累计坏账率": np.nan})
                continue
            dq = delinq_months_at(g, m)
            rows.append({"cohort": c, "mob": m, "n": n,
                         "累计坏账率": float((dq >= months_delinq).mean())})
    return pd.DataFrame(rows)
