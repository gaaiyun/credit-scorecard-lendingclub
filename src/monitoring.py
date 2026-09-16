# -*- coding: utf-8 -*-
"""上线后监控：把模型件读回来给新批次打分，算 PSI / CSI / KS 衰减并判定动作。

**监控最反直觉的一点：指标不是同时可得的**
分数 PSI、变量 CSI、审批率、分数均值 —— 放款当月就能算。
KS、vintage 坏账率 —— 要等一整个表现期（本项目 18 个月）才有标签。
也就是说：模型效果真正掉下来的时候，你要等 18 个月才能用 KS 证明它掉了。
所以监控体系必须靠**前置指标**（PSI/CSI）先报警，KS 是事后确认，不是触发条件。
这一点在设计监控方案时经常被忽略，做出来就成了一套"事后诸葛亮"的看板。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scorecard import (apply_bins, apply_cat_bins, csi, ks_auc,  # noqa: F401
                       psi, to_score)


def load_scorer(model_json: str | Path):
    """把 step3 存下来的模型件读回来，返回 (打分函数, 打箱函数, 入模变量)。

    这一步等价于生产环境的"模型加载"：评分卡上线部署的就是这么一个
    切点 + WOE 映射 + 系数 的 JSON，不需要带 python 对象或 pickle。
    这也是评分卡相对树模型的一个实际优势——模型件是人能读懂的表。
    """
    art = json.loads(Path(model_json).read_text(encoding="utf-8"))
    keep = set(art["final_feats"])
    # 模型件里存了全部 70 个变量的分箱（留档用），但上线只需要入模的那 16 个。
    # 只打这 16 个变量的箱，内存和耗时都省一大截。
    num_cuts = {k: list(v) for k, v in art["num_cuts"].items() if k in keep}
    cat_maps = {k: {kk: int(vv) for kk, vv in v.items()}
                for k, v in art["cat_maps"].items() if k in keep}
    wmap = {}
    for k, v in art["woe_map"].items():
        var, b = k.rsplit("||", 1)
        wmap[(var, int(b))] = float(v)
    feats = art["final_feats"]
    params = art["params"]
    A, B = art["A"], art["B"]

    def bins_of(d: pd.DataFrame) -> pd.DataFrame:
        out = {f: apply_bins(d[f], c) for f, c in num_cuts.items() if f in d.columns}
        out.update({f: apply_cat_bins(d[f], m)
                    for f, m in cat_maps.items() if f in d.columns})
        return pd.DataFrame(out, index=d.index)

    def score_of(d: pd.DataFrame) -> np.ndarray:
        b = bins_of(d)
        lin = np.full(len(d), float(params["const"]))
        for f in feats:
            lin = lin + float(params[f]) * b[f].map(
                lambda k, f=f: wmap.get((f, int(k)), 0.0)).to_numpy(dtype=float)
        p = 1.0 / (1.0 + np.exp(-lin))
        return to_score(p, A, B)

    return score_of, bins_of, feats, art


def verdict(value: float, watch: float, alert: float) -> str:
    if value >= alert:
        return "报警"
    if value >= watch:
        return "关注"
    return "正常"


def monthly_monitor(d: pd.DataFrame, month_col: str, score: np.ndarray,
                    bins: pd.DataFrame, base_score: np.ndarray,
                    base_bins: pd.DataFrame, feats: list[str],
                    y: pd.Series | None = None,
                    observable: pd.Series | None = None,
                    psi_watch=0.10, psi_alert=0.25,
                    csi_watch=0.10, csi_alert=0.25) -> pd.DataFrame:
    """逐月监控报表。

    base_*  建模期（训练集）的基准分布
    y / observable  标签与"该月是否已走完表现期"。没走完的月份 KS 一栏留空，
                    这正是生产上的真实情况：新月份只有前置指标，没有效果指标。
    """
    rows = []
    for m, idx in d.groupby(month_col, observed=True).groups.items():
        pos = d.index.get_indexer(idx)
        s = score[pos]
        p = psi(base_score, s)
        c_max, c_var = 0.0, ""
        for f in feats:
            v = csi(base_bins[f], bins[f].iloc[pos])
            if v > c_max:
                c_max, c_var = v, f
        row = {"月份": str(m), "笔数": len(idx),
               "分数均值": round(float(s.mean()), 2),
               "分数PSI": round(p, 4), "PSI判定": verdict(p, psi_watch, psi_alert),
               "最大CSI": round(c_max, 4), "最大CSI变量": c_var,
               "CSI判定": verdict(c_max, csi_watch, csi_alert)}
        if y is not None and observable is not None and bool(observable.loc[idx].all()):
            yy = y.loc[idx]
            if yy.nunique() > 1:
                ks, auc = ks_auc(yy, s)
                row["KS"] = round(ks, 4)
                row["坏客户率"] = round(float(yy.mean()), 4)
        rows.append(row)
    return pd.DataFrame(rows).sort_values("月份").reset_index(drop=True)


def action_for(psi_v: float, csi_v: float, ks_drop: float | None,
               psi_watch=0.10, psi_alert=0.25,
               csi_watch=0.10, csi_alert=0.25,
               ks_watch=0.10, ks_alert=0.20) -> str:
    """把指标映射成动作。监控方案的价值在这一列——只报数不给动作等于没做监控。"""
    if ks_drop is not None and ks_drop >= ks_alert:
        return "立即重训：效果已确认劣化，同时冻结 cutoff 下调申请"
    if psi_v >= psi_alert:
        return "人群已显著漂移：先查是渠道变了还是口径变了，确认后重新分箱或重训"
    if csi_v >= csi_alert:
        return "单变量漂移超阈：核查该变量的数据源与采集口径，必要时剔除该变量重训"
    if ks_drop is not None and ks_drop >= ks_watch:
        return "效果衰减进入关注区：排查最近的政策/渠道变更，准备重训样本"
    if psi_v >= psi_watch or csi_v >= csi_watch:
        return "关注：加密到每周监控，同时检查审批率与通过客户结构是否同步变化"
    return "正常：保持月度监控"
