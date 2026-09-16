# -*- coding: utf-8 -*-
"""收益测算：把评分卡的排序能力折算成钱，用来定 cutoff。

**所有假设集中在这里，不许散落在计算过程里**
风控面试最常见的追问是「你这个 cutoff 凭什么定在这儿」。
答「坏账率不超过基准八成」是技术答案，不是业务答案——业务要的是
单位放款的净收益最大，或者在给定风险偏好下的审批量最大。
所以必须把收入、损失、资金成本三块拆开，每一块的假设单独列。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd


@dataclass
class Assumptions:
    """收益测算的全部假设。每一条都要能说出依据或标明是拍的。"""

    lgd: float = 0.8917
    lgd_source: str = ("实测：全量 Charged Off 样本 sum(recoveries)/sum(未收回本金) "
                       "= 10.83%，故 LGD = 1 - 10.83% = 89.17%（step6 会重算一遍）")

    funding_cost: float = 0.04
    funding_cost_source: str = "假设：年化 4%。2016-2017 年美国消费信贷市场的大致资金成本"

    opex_rate: float = 0.01
    opex_rate_source: str = "假设：年化 1% 的获客与运营成本摊到放款余额上"

    avg_life_years: float = 1.6
    avg_life_source: str = ("假设：36 期等额本息、考虑提前结清，平均存续期约 1.6 年。"
                            "教科书口径 1.5~1.8 年，取中值")

    ead_ratio: float = 0.75
    ead_source: str = ("假设：违约时点的风险敞口 = 放款金额 x 75%。"
                       "36 期贷款在 MOB 12~18 违约时约还掉四分之一本金")

    def table(self) -> pd.DataFrame:
        rows = [
            ("LGD 违约损失率", f"{self.lgd:.4f}", self.lgd_source),
            ("资金成本（年化）", f"{self.funding_cost:.2%}", self.funding_cost_source),
            ("运营成本（年化）", f"{self.opex_rate:.2%}", self.opex_rate_source),
            ("平均存续期（年）", f"{self.avg_life_years:.2f}", self.avg_life_source),
            ("违约时敞口比例 EAD", f"{self.ead_ratio:.2f}", self.ead_source),
        ]
        return pd.DataFrame(rows, columns=["假设项", "取值", "依据 / 是否为拍定值"])


def unit_economics(amount: np.ndarray, int_rate_pct: np.ndarray,
                   y: np.ndarray, a: Assumptions) -> pd.DataFrame:
    """逐笔算收入与损失。

    好客户：利息收入 = 本金 x 年化利率 x 平均存续期
    坏客户：损失 = 本金 x EAD比例 x LGD，同时按违约前已收的部分利息折半计
    两类都要扣资金成本与运营成本

    这是一个**简化模型**：没有做现金流贴现，没有区分违约发生的时点，
    也没有考虑提前结清带来的利息损失。它够用来在不同 cutoff 之间做相对比较，
    不够用来做真实定价。这一点报告里要写明。
    """
    amt = np.asarray(amount, dtype=float)
    r = np.asarray(int_rate_pct, dtype=float) / 100.0
    y = np.asarray(y)

    carry = amt * (a.funding_cost + a.opex_rate) * a.avg_life_years
    interest_good = amt * r * a.avg_life_years
    # 坏客户在违约前也还过几期，按平均存续期的一半计利息
    interest_bad = amt * r * (a.avg_life_years * 0.5)
    loss_bad = amt * a.ead_ratio * a.lgd

    revenue = np.where(y == 1, interest_bad, interest_good)
    loss = np.where(y == 1, loss_bad, 0.0)
    return pd.DataFrame({"放款金额": amt, "收入": revenue,
                         "损失": loss, "资金运营成本": carry,
                         "净收益": revenue - loss - carry, "y": y})


def profit_by_cutoff(score: np.ndarray, econ: pd.DataFrame,
                     n_steps: int = 40) -> pd.DataFrame:
    """不同 cutoff 下的审批量、坏账率与收益。

    逐个分位点扫过去，算「分数 >= 该点」时的总净收益、单位放款净收益、
    通过率与坏账率。推荐 cutoff 就是在这张表上挑。
    """
    s = np.asarray(score, dtype=float)
    qs = np.linspace(0, 97.5, n_steps)
    rows = []
    for q in qs:
        thr = np.percentile(s, q)
        m = s >= thr
        if m.sum() < 100:
            continue
        e = econ[m]
        n = int(m.sum())
        rows.append({
            "cutoff分数": round(float(thr), 1),
            "通过率": round(float(m.mean()), 4),
            "通过笔数": n,
            "坏账率": round(float(e["y"].mean()), 4),
            "放款总额": round(float(e["放款金额"].sum()), 0),
            "净收益": round(float(e["净收益"].sum()), 0),
            "单位放款净收益": round(float(e["净收益"].sum() / e["放款金额"].sum()), 5),
            "件均净收益": round(float(e["净收益"].mean()), 2),
        })
    return pd.DataFrame(rows)


def empirical_lgd(funded: np.ndarray, rec_prncp: np.ndarray,
                  recoveries: np.ndarray) -> tuple[float, float, float]:
    """用数据里真实的核销回收反算 LGD，校准假设。

    这三个字段都是放款后产生的，**不能进模型**（见 leakage.py 的判定），
    但拿来校准 LGD 假设是合理的——那是事后的损失核算，不是事前的预测特征。
    """
    lost = np.clip(np.asarray(funded, float) - np.asarray(rec_prncp, float), 0, None)
    rec = np.nan_to_num(np.asarray(recoveries, float))
    rr = float(rec.sum() / max(lost.sum(), 1.0))
    return 1.0 - rr, rr, float(lost.sum())
