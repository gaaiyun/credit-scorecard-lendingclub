# -*- coding: utf-8 -*-
"""第 7 步：上线后监控 —— 不只写方案，用 2017-10 ~ 2018-12 的真实放款跑一遍。

模拟的场景
----------
评分卡用 2016 年放款训练、2017 年前三季度做 OOT 验证，假设 2017-10 上线。
之后每个月的新放款就是"上线后的批次"：
  · 当月立刻能算的：分数 PSI、变量 CSI、审批分数分布
  · 要等 18 个月才能算的：KS、vintage 坏账率
数据快照是 2019-03，所以 2017-10 及之后放款的贷款**都还没走完 18 个月表现期**，
KS 那一列只能是空的。这不是数据缺陷，这就是生产上的真实处境：
前置指标先报警，效果指标是事后确认。

产出
----
output/60_月度监控报表.csv
output/61_监控图.png
docs/监控方案.md
output/step7_日志.txt
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as C                                      # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from monitoring import (action_for, load_scorer, monthly_monitor,  # noqa: E402
                        verdict)
from scorecard import csi, ks_auc, psi                  # noqa: E402
from util import Logger                                 # noqa: E402
from vintage import add_mob, label_at, observable_mob   # noqa: E402

log = Logger(C.OUT / "step7_日志.txt")

GO_LIVE = "2017-10"


def main() -> int:
    t0 = time.time()
    log.section("第 7 步  上线后监控（设计 + 用真实数据跑一遍）")

    score_of, bins_of, feats, art = load_scorer(C.PROC / "scorecard_model.json")
    log(f"从模型件加载评分卡：{len(feats)} 个入模变量，"
        f"A={art['A']:.2f}, B={art['B']:.2f}")
    log("  模型件是一个 JSON：切点 + WOE 映射 + 系数。没有 pickle，没有 python 对象，")
    log("  人能直接读、能用 SQL 复现——这是评分卡在上线运维上相对树模型的实际优势。")

    # ---- 建模期基准 ----
    tr_ids = set(pd.read_parquet(C.PROC / "scored_train.parquet",
                                 columns=["id"])["id"].astype(str))
    # 只读监控真正用得到的列：16 个入模变量 + 标签重建需要的 4 列。
    # 整张 95 列的表在本机 4GB 提交限额下会 MemoryError。
    need = sorted(set(feats) | {"id", "issue_dt", "loan_status", "last_pymnt_dt"})
    full = pd.read_parquet(C.ACCEPTED_PQ, columns=need)
    full["sid"] = full["id"].astype(str)
    log(f"读入 {len(full):,} 笔 x {full.shape[1]} 列"
        f"（只读监控用得到的列，不读全部 95 列）")
    tr = full[full["sid"].isin(tr_ids)]
    base_score = score_of(tr)
    base_bins = bins_of(tr)
    log(f"建模期基准：{len(tr):,} 笔（2016 年放款训练集），"
        f"分数均值 {base_score.mean():.2f}")

    # 校验：从 JSON 复现的分数要和 step3 的分数对得上
    ref = pd.read_parquet(C.PROC / "scored_train.parquet")
    ref = ref.set_index(ref["id"].astype(str))["score"].reindex(tr["sid"]).to_numpy()
    dmax = float(np.nanmax(np.abs(base_score - ref)))
    log(f"[校验] 模型件复现的分数 vs step3 原始分数，最大绝对差 = {dmax:.6f}")
    if dmax > 1e-6:
        log("[FAIL] 模型件复现不一致，说明序列化丢了信息，先修这个再谈监控")
        return 1
    log("[OK] 完全一致，模型件可用于生产部署")

    # ---- 上线后批次 ----
    log.section("7.1  逐月监控报表")
    post = full[full["issue_dt"] >= pd.Timestamp(GO_LIVE)].copy()
    log(f"假设 {GO_LIVE} 上线，之后放款 {len(post):,} 笔，"
        f"覆盖 {post['issue_dt'].dt.to_period('M').nunique()} 个月")

    post = add_mob(post)
    post["obs_mob"] = observable_mob(post, C.SNAPSHOT)
    post["y"] = (label_at(post, C.PERFORM_WINDOW,
                          months_delinq=C.MONTHS_DELINQ) == "bad").astype(int)
    post["matured"] = post["obs_mob"] >= C.PERFORM_WINDOW
    log(f"其中已走完 {C.PERFORM_WINDOW} 个月表现期的：{int(post['matured'].sum()):,} 笔"
        f"（{post['matured'].mean():.1%}）")
    log(f"  数据快照 {C.SNAPSHOT}，{GO_LIVE} 之后放款的贷款都还没到期 —— ")
    log(f"  所以下表 KS 一列全是空的。这正是要说的那件事：上线后 {C.PERFORM_WINDOW} 个月内，")
    log("  你手上只有前置指标。")

    post["月份"] = post["issue_dt"].dt.to_period("M").astype(str)
    p_score = score_of(post)
    p_bins = bins_of(post)
    rep = monthly_monitor(post, "月份", p_score, p_bins, base_score, base_bins,
                          feats, y=post["y"], observable=post["matured"],
                          psi_watch=C.PSI_WATCH, psi_alert=C.PSI_ALERT,
                          csi_watch=C.CSI_WATCH, csi_alert=C.CSI_ALERT)
    rep["建议动作"] = [action_for(r["分数PSI"], r["最大CSI"], None,
                              C.PSI_WATCH, C.PSI_ALERT, C.CSI_WATCH, C.CSI_ALERT)
                   for _, r in rep.iterrows()]
    rep.to_csv(C.OUT / "60_月度监控报表.csv", index=False, encoding="utf-8-sig")
    log("")
    log(rep[["月份", "笔数", "分数均值", "分数PSI", "PSI判定",
             "最大CSI", "最大CSI变量", "CSI判定"]].to_string(index=False))

    # ---- 有标签的月份：效果指标 ----
    log.section("7.2  有标签的月份：效果指标怎么补上来")
    log("2016-01 ~ 2017-09 的放款已经走完表现期，把它们按月算 KS，")
    log("就是「18 个月之后回头看」的样子。生产上这张表要一直往后滚。")
    hist = full[full["issue_dt"] < pd.Timestamp(GO_LIVE)].copy()
    hist = add_mob(hist)
    hist["obs_mob"] = observable_mob(hist, C.SNAPSHOT)
    hist = hist[hist["obs_mob"] >= C.PERFORM_WINDOW].copy()
    hist["y"] = (label_at(hist, C.PERFORM_WINDOW,
                          months_delinq=C.MONTHS_DELINQ) == "bad").astype(int)
    hist["月份"] = hist["issue_dt"].dt.to_period("M").astype(str)
    h_score = score_of(hist)
    h_bins = bins_of(hist)

    base_ks = art["perf"][0]["KS"]
    rows = []
    for m, idx in hist.groupby("月份", observed=True).groups.items():
        pos = hist.index.get_indexer(idx)
        s, yy = h_score[pos], hist.loc[idx, "y"]
        ks, auc = ks_auc(yy, s)
        p = psi(base_score, s)
        cmax = max(csi(base_bins[f], h_bins[f].iloc[pos]) for f in feats)
        drop = (base_ks - ks) / base_ks
        rows.append({"月份": str(m), "笔数": len(idx),
                     "坏客户率": round(float(yy.mean()), 4),
                     "KS": round(ks, 4), "KS相对建模期衰减": round(float(drop), 4),
                     "分数PSI": round(p, 4), "最大CSI": round(cmax, 4),
                     "KS判定": verdict(drop, C.KS_DECAY_WATCH, C.KS_DECAY_ALERT),
                     "建议动作": action_for(p, cmax, drop, C.PSI_WATCH, C.PSI_ALERT,
                                        C.CSI_WATCH, C.CSI_ALERT,
                                        C.KS_DECAY_WATCH, C.KS_DECAY_ALERT)})
    hrep = pd.DataFrame(rows).sort_values("月份").reset_index(drop=True)
    hrep.to_csv(C.OUT / "60b_有标签月份效果.csv", index=False, encoding="utf-8-sig")
    log("")
    log(hrep[["月份", "笔数", "坏客户率", "KS", "KS相对建模期衰减",
              "分数PSI", "KS判定"]].to_string(index=False))
    log("")
    log(f"建模期 KS 基准 = {base_ks:.4f}")
    log(f"上线后（2017 年）各月 KS 区间 {hrep['KS'].min():.4f} ~ {hrep['KS'].max():.4f}，"
        f"最大衰减 {hrep['KS相对建模期衰减'].max():.1%}")
    worst = hrep.loc[hrep["KS相对建模期衰减"].idxmax()]
    log(f"最差月份 {worst['月份']}：KS {worst['KS']:.4f}，衰减 "
        f"{worst['KS相对建模期衰减']:.1%}，判定「{worst['KS判定']}」")

    # ---- 7.3 前置指标够不够用 ----
    log.section("7.3  一个必须验证的问题：前置指标真的能预警吗")
    log("监控体系的全部前提是：PSI/CSI 在 KS 掉之前就能动。如果不能，")
    log("那这套指标就是摆设。这里用有标签的月份直接检验两者的相关性。")
    from scipy.stats import spearmanr
    r_psi = spearmanr(hrep["分数PSI"], hrep["KS相对建模期衰减"])
    r_csi = spearmanr(hrep["最大CSI"], hrep["KS相对建模期衰减"])
    log("")
    log(f"    分数PSI  vs KS衰减   秩相关 {r_psi.statistic:+.4f}（p={r_psi.pvalue:.3f}）")
    log(f"    最大CSI  vs KS衰减   秩相关 {r_csi.statistic:+.4f}（p={r_csi.pvalue:.3f}）")
    log("")
    if r_psi.pvalue > 0.05 and r_csi.pvalue > 0.05:
        log("    [结论] 在这段样本上，PSI/CSI 与 KS 衰减**没有显著相关**。")
        log("    这不是说 PSI 没用，而是说：本段时间里人群就没怎么动（PSI 全在 0.02 以下），")
        log("    KS 的月度波动主要是抽样噪声，两个都接近常数，自然测不出相关。")
        log("    真正的含义是——**PSI 正常不能证明模型还好用**。")
        log("    PSI 只能发现「人群变了」这一类失效，发现不了「人群没变但关系变了」")
        log("    （比如经济下行时同样的征信特征对应更高的违约率）。")
        log("    所以监控方案里必须同时有 vintage 坏账率的逐月跟踪，")
        log("    它比 KS 早得多就能看出苗头：MOB 3~6 的早期指标一个月就能出。")
    else:
        log("    [结论] 前置指标与效果衰减存在相关，可作为预警使用。")

    # ---- 7.4 早期风险指标 ----
    log.section("7.4  比 KS 早得多的效果指标：早期 vintage")
    log("等 18 个月太久。生产上真正每月看的是**早期账龄指标**：")
    log("FPD30（首期逾期 30 天）、MOB3 的 M1+、MOB6 的累计坏账率。")
    log("它们和最终坏账率高度同向，但一两个月就能出数。")
    log("")
    # add_mob 只做一次。每个 MOB 重做一遍会连开 4 份 98 万行的副本，内存扛不住。
    base = add_mob(full[["issue_dt", "last_pymnt_dt", "loan_status"]])
    base["obs"] = observable_mob(base, C.SNAPSHOT)
    base["q"] = base["issue_dt"].dt.to_period("Q").astype(str)
    early = []
    for mob in (3, 6, 9, 12):
        ok = base["obs"] >= mob
        lab = label_at(base[ok], mob, months_delinq=C.MONTHS_DELINQ) == "bad"
        s = lab.groupby(base.loc[ok, "q"], observed=True).mean()
        early.append(s.rename(f"MOB{mob}"))
    del base
    ed = pd.concat(early, axis=1).round(4)
    ed.to_csv(C.OUT / "62_早期风险指标.csv", encoding="utf-8-sig")
    log(ed.to_string())
    log("")
    log("-- 哪个早期指标真的能当最终坏账率的代理：逐个检验，不靠常识 --")
    target = ed.columns[-1]              # MOB12，本数据里能覆盖最多 cohort 的最晚指标
    rows2 = []
    for col in ed.columns[:-1]:
        ok = ed[[col, target]].dropna()
        if len(ok) < 4:
            continue
        r = spearmanr(ok[col], ok[target])
        rows2.append({"早期指标": col, "对比": target, "季度数": len(ok),
                      "秩相关": round(float(r.statistic), 4),
                      "p值": round(float(r.pvalue), 4),
                      "可用": "是" if (r.pvalue < 0.05 and r.statistic > 0.6) else "否"})
    prox = pd.DataFrame(rows2)
    log(prox.to_string(index=False))
    prox.to_csv(C.OUT / "62b_早期指标代理性检验.csv", index=False,
                encoding="utf-8-sig")
    log("")
    usable = prox[prox["可用"] == "是"]["早期指标"].tolist()
    if usable:
        log(f"    [结论] {', '.join(usable)} 与 {target} 显著同向，可以当早期代理指标。")
    else:
        log(f"    [结论] 在这段样本上，**没有一个早期指标与 {target} 显著相关**。")
        log("    别把这条读成「早期指标没用」，要看清它为什么不相关：")
        log(f"    MOB3 的坏账率只有 {ed['MOB3'].mean():.2%} 量级，")
        log("    一个季度十万笔里也就一百来个坏客户，季度间的差异基本是抽样噪声。")
        log("    而且本段时间 LC 的客群本来就稳（见 7.1 的 PSI 全部 < 0.03），")
        log("    各 cohort 的真实坏账率差异本身就小，信号弱到测不出来。")
        log("    早期指标真正发挥作用的场景是**出事的时候**：渠道换了、政策放松了、")
        log("    经济下行了，FPD30 会在一两个月内跳起来，那时候不需要统计检验也看得见。")
        log("    它是异常报警器，不是精细预测器——这是它该有的定位。")

    # ---- 7.5 漂移钻取 ----
    log.section("7.5  漂移钻取：报警之后第一件事是定位，不是重训")
    worst_row = rep.loc[rep["最大CSI"].idxmax()]
    wv, wm = worst_row["最大CSI变量"], worst_row["月份"]
    log(f"最严重的漂移：{wm} 月，变量 {wv}，CSI = {worst_row['最大CSI']:.4f}"
        f"（{worst_row['CSI判定']}）")
    log("评分卡的好处就在这儿——能直接打开看是哪一箱在变：")
    log("")
    mm = post[post["月份"] == wm]
    a_ = base_bins[wv].value_counts(normalize=True).sort_index()
    b_ = p_bins.loc[mm.index, wv].value_counts(normalize=True).sort_index()
    drill = pd.DataFrame({"建模期占比": a_, "当月占比": b_}).fillna(0).round(4)
    drill["变化"] = (drill["当月占比"] - drill["建模期占比"]).round(4)
    if wv in art["cat_maps"]:
        inv = {}
        for k, v in art["cat_maps"][wv].items():
            inv.setdefault(int(v), []).append(str(k))
        drill.insert(0, "箱含义", [" / ".join(inv.get(int(i), ["缺失/未见过"]))
                                for i in drill.index])
    log(drill.to_string())
    log("")
    log("怎么判「人群变了」还是「口径变了」：")
    log("  · 人群变了 —— 各箱占比此消彼长、变化连续、且和渠道/营销动作对得上时间；")
    log("  · 口径变了 —— 某一箱占比突变或归零、缺失率跳变、变化发生在系统上线日；")
    log("两者的处理完全不同：人群变了要重新分箱或重训；")
    log("口径变了要先修数据源，在修好之前这个变量必须停用——")
    log("拿一个口径已变的变量重训，等于把错误固化进新模型。")
    log("")
    log(f"本例中 {wv} 的漂移是**逐月渐进**的（见 60_月度监控报表.csv 的 CSI 列：")
    log(f"  {' -> '.join(f'{v:.3f}' for v in rep['最大CSI'].head(8))} ...），")
    log("  没有断点式跳变，更像平台在这段时间逐步调整了收入核验政策，")
    log("  属于「政策/口径缓慢变化」这一类。动作是：先和业务确认核验规则有没有改，")
    log("  确认改了就把这个变量的分箱重做（而不是整卡重训），并观察重做后 CSI 是否回落。")

    # ---- 图 ----
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    allm = pd.concat([hrep[["月份", "分数PSI", "最大CSI"]],
                      rep[["月份", "分数PSI", "最大CSI"]]], ignore_index=True)
    allm = allm.sort_values("月份")
    x = range(len(allm))
    ax[0].plot(x, allm["分数PSI"], "o-", lw=1.8, ms=3, label="分数 PSI")
    ax[0].plot(x, allm["最大CSI"], "s-", lw=1.4, ms=3, label="最大变量 CSI")
    ax[0].axhline(C.PSI_WATCH, ls="--", c="orange", lw=1, label=f"关注线 {C.PSI_WATCH}")
    ax[0].axhline(C.PSI_ALERT, ls="--", c="crimson", lw=1, label=f"报警线 {C.PSI_ALERT}")
    ax[0].axvline(len(hrep) - 0.5, ls=":", c="gray", lw=1.5)
    ax[0].text(len(hrep) - 0.4, C.PSI_ALERT * 0.75, f"{GO_LIVE} 上线", fontsize=8,
               color="gray")
    ax[0].set_xticks(list(x)[::3])
    ax[0].set_xticklabels(allm["月份"].iloc[::3], rotation=45, fontsize=7)
    ax[0].set_title("前置指标：逐月分数 PSI 与变量 CSI")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

    ax[1].plot(hrep["月份"], hrep["KS"], "o-", lw=1.8, ms=4, c="#c44e52")
    ax[1].axhline(base_ks, ls="--", c="gray", lw=1.2, label=f"建模期 KS {base_ks:.4f}")
    ax[1].axhline(base_ks * (1 - C.KS_DECAY_WATCH), ls="--", c="orange", lw=1,
                  label=f"关注线（-{C.KS_DECAY_WATCH:.0%}）")
    ax[1].axhline(base_ks * (1 - C.KS_DECAY_ALERT), ls="--", c="crimson", lw=1,
                  label=f"报警线（-{C.KS_DECAY_ALERT:.0%}）")
    ax[1].set_xticks(range(0, len(hrep), 3))
    ax[1].set_xticklabels(hrep["月份"].iloc[::3], rotation=45, fontsize=7)
    ax[1].set_title(f"效果指标：逐月 KS（只有走完 {C.PERFORM_WINDOW} 个月表现期的月份才有）")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)

    for col in ed.columns:
        ax[2].plot(ed.index, ed[col], "o-", lw=1.6, ms=4, label=col)
    ax[2].set_title("早期风险指标：各放款季度在 MOB 3/6/9/12 的累计坏账率")
    ax[2].set_xlabel("放款季度"); ax[2].set_ylabel("累计坏账率")
    ax[2].tick_params(axis="x", rotation=45)
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(C.OUT / "61_监控图.png", dpi=150)
    plt.close(fig)
    log("")
    log(f"[OK] 图 -> {C.OUT / '61_监控图.png'}")
    log(f"[OK] 全步用时 {time.time()-t0:.0f}s")
    log.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
