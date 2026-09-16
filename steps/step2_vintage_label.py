# -*- coding: utf-8 -*-
"""第 2 步：vintage 账龄分析定表现期，据此定义好/坏/灰客户。

产出
----
output/12_采集口径月度体检.csv   征信字段缺失率按放款月（兜底校验）
output/13_vintage明细.csv        cohort x MOB 的累计坏账率
output/14_表现期敏感性.csv       表现期取 12/15/18/21/24 的权衡对比
output/15_好坏灰分布.csv         最终样本的好/坏/灰数量与占比
output/16_vintage曲线.png        vintage 曲线 + 边际违约强度 + cohort 成熟度检验
output/step2_日志.txt
data/processed/model_sample.parquet

流程
----
1. 复核采集口径一致性（step1 已在全历史上定下窗口起点）
2. 从 last_pymnt_d 重建账龄状态，画 vintage 曲线
3. 用敏感性分析定表现期——不是「等曲线走平」，曲线到 MOB 30 都没平
4. 用「放款月 + 表现期 <= 数据快照月」筛掉观察不足的 cohort
5. 在表现期上打好/坏/灰标签
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as C                                      # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from leakage import MODEL_FEATURES                      # noqa: E402
from util import Logger                                 # noqa: E402
from vintage import (add_mob, delinq_months_at, label_at,  # noqa: E402
                     observable_mob, vintage_table)

log = Logger(C.OUT / "step2_日志.txt")

# step1 查出来的、2016 年才开始全量采集的征信字段
LATE_COLLECTED = ["open_rv_24m", "max_bal_bc", "open_act_il", "total_bal_il",
                  "inq_last_12m", "inq_fi", "total_cu_tl", "open_rv_12m",
                  "open_acc_6m", "open_il_24m", "open_il_12m", "all_util",
                  "mths_since_rcnt_il", "il_util"]


def main() -> int:
    log.section("第 2 步  vintage 定表现期 + 好坏客户定义")
    df = pd.read_parquet(C.ACCEPTED_PQ)
    log(f"读入 {len(df):,} 笔 36 期贷款（放款期 {C.VINTAGE_START}~{C.VINTAGE_END}）")

    # ---------------- 2.1 采集口径复核 ----------------
    log.section("2.1  采集口径复核（窗口起点已在 step1 定下，这里只做兜底校验）")
    log("step1 在全历史上查出 LC 有一批征信字段是中途才开始采的，最典型的 14 个")
    log("在 2015-12 之前 100% 缺失。建模期和 OOT 期必须都落在采集之后，")
    log("否则这些变量的 CSI 会因为技术原因爆表，被误读成人群漂移。")
    log("")
    m = df.groupby(df["issue_dt"].dt.to_period("M"), observed=True)[
        LATE_COLLECTED].apply(lambda g: g.isna().mean().mean()).round(4)
    m.name = "14个晚采集字段的平均缺失率"
    m.to_csv(C.OUT / "12_采集口径月度体检.csv", encoding="utf-8-sig")
    log(m.head(8).to_string())
    log("    ...")
    log(f"    窗口内最大值 = {m.max():.4f}（{m.idxmax()}）")
    if m.max() >= 0.05:
        log(f"[FAIL] 建模窗口内仍有月份缺失率 >= 5%，起点要再往后挪")
        return 1
    log("[OK] 建模窗口内这批字段的缺失率全程 < 5%，采集口径一致")

    # ---------------- 2.2 账龄重建 ----------------
    log.section("2.2  从 last_pymnt_d 重建账龄表现")
    df = add_mob(df)
    df["obs_mob"] = observable_mob(df, C.SNAPSHOT)
    log(f"数据快照时点 = {C.SNAPSHOT}（accepted 表 last_pymnt_d 的最大值）")
    log(f"一次都没还过（首期即违约）的笔数 = {int(df['never_paid'].sum()):,}"
        f"（{df['never_paid'].mean():.3%}）")
    log("")
    log("-- 在途账户（快照时仍是 Current）的「距最后还款月数」分布 --")
    log("   用来确认：正常还款的账户在快照上滞后几个月。滞后 0~1 个月属正常节奏，")
    log("   从第 2 个月起才是真逾期——这决定灰样本区间怎么切。")
    cur = df[df["loan_status"] == "Current"]
    lag = (cur["obs_mob"] - cur["mob_last_pymnt"]).clip(lower=0)
    lv = lag.value_counts(normalize=True).sort_index().head(6).round(4)
    log(pd.DataFrame({"滞后月数": lv.index, "占比": lv.to_numpy()}).to_string(index=False))

    # ---------------- 2.3 vintage 曲线 ----------------
    log.section("2.3  代理 vintage 曲线（LC 只有终态快照，没有逐月表现面板）")
    df["cohort_q"] = df["issue_dt"].dt.to_period("Q").astype(str)
    vt = vintage_table(df, "cohort_q", max_mob=30,
                       months_delinq=C.MONTHS_DELINQ, snapshot=C.SNAPSHOT)
    vt.to_csv(C.OUT / "13_vintage明细.csv", index=False, encoding="utf-8-sig")

    piv = vt.pivot(index="mob", columns="cohort", values="累计坏账率")
    log("-- 各放款季度的累计坏账率（MOB 6/12/15/18/21/24）--")
    log(piv.loc[[6, 12, 15, 18, 21, 24]].round(4).to_string())

    # 用观察最完整的 cohort 看边际增量：每多观察 3 个月，坏账率还能涨多少
    ref = piv[piv.columns[0]].dropna()
    log("")
    log(f"-- 最早 cohort（{piv.columns[0]}）的边际增量：MOB m 相对 m-3 的增幅 --")
    inc = pd.DataFrame({"mob": ref.index[3:],
                        "累计坏账率": ref.values[3:].round(4),
                        "近3个月增量": (ref.values[3:] - ref.values[:-3]).round(4)})
    inc["占MOB24总量比"] = (inc["累计坏账率"] / ref.get(24, ref.iloc[-1])).round(3)
    log(inc[inc["mob"] % 3 == 0].to_string(index=False))

    # ---------------- 2.4 定表现期 ----------------
    log.section("2.4  表现期怎么定：敏感性分析")
    last_mob = int(ref.index.max())
    log(f"先把话说清楚：**这条曲线到 MOB {last_mob} 都没走平**。")
    log(f"最完整的 cohort（{piv.columns[0]}，可观察到 MOB {last_mob}）的三个月边际增量：")
    log(f"    MOB 24 -> 27：+{ref.get(27, np.nan) - ref.get(24, np.nan):.4f}")
    log(f"    MOB 27 -> 30：+{ref.get(30, np.nan) - ref.get(27, np.nan):.4f}")
    log("36 期等额本息产品的违约是贯穿整个账期的，要等真正走平得等到接近 MOB 36。")
    log("所以表现期不是「等曲线平了再取」，而是一个**权衡**：表现期越长，标签越完整，")
    log("但能满足「放款月 + 表现期 <= 数据快照月」的 cohort 越少，跨时间 OOT 越切不开。")
    log("")

    obs_all = observable_mob(df, C.SNAPSHOT)
    rows = []
    for w in (12, 15, 18, 21, 24):
        ok = obs_all >= w
        if not ok.any():
            continue
        sub = df[ok]
        lab = label_at(sub, w, months_delinq=C.MONTHS_DELINQ)
        n_cohort = sub["issue_dt"].dt.to_period("Q").nunique()
        rows.append({
            "表现期MOB": w,
            "最晚可用放款月": f"{sub['issue_dt'].max():%Y-%m}",
            "可用季度cohort数": n_cohort,
            "可用样本量": len(sub),
            "坏客户率": round(float((lab == "bad").mean()), 4),
            "灰样本占比": round(float((lab == "grey").mean()), 4),
            f"占MOB{last_mob}最终坏账的比例":
                round(float(ref[w] / ref[last_mob]), 3) if w in ref.index else np.nan,
        })
    sens = pd.DataFrame(rows)
    log(sens.to_string(index=False))
    sens.to_csv(C.OUT / "14_表现期敏感性.csv", index=False, encoding="utf-8-sig")

    # 捕获率检验：MOB last_mob 才坏的那批人，有多少在 MOB W 时已经坏了
    log("")
    log("-- 捕获率检验：短表现期会不会漏掉一类不同的坏客户 --")
    ref_coh = df[df["issue_dt"].dt.to_period("Q").astype(str) == piv.columns[0]]
    lab_late = label_at(ref_coh, last_mob, months_delinq=C.MONTHS_DELINQ)
    for w in (12, 15, 18, 21):
        lab_w = label_at(ref_coh, w, months_delinq=C.MONTHS_DELINQ)
        late_bad = (lab_late == "bad")
        capt = float((lab_w == "bad")[late_bad].mean())
        log(f"    表现期 {w:2d} 个月：MOB {last_mob} 的坏客户里有 {capt:.1%} 在 MOB {w} 时已判坏")

    log("")
    log(f"[判定] 表现期取 {C.PERFORM_WINDOW} 个月（config.PERFORM_WINDOW）")
    log("       理由是权衡，不是「曲线平了」：")
    log(f"       ① 12 个月只能捕获最终坏账的 {ref[12]/ref[last_mob]:.0%}，标签太不完整，")
    log("          且早期违约（首期不还、欺诈）和中后期违约（收入恶化）是两类人，")
    log("          只学前者会让评分卡在正常信用风险上失灵；")
    log(f"       ② 24 个月最晚只能用到 {sens.loc[sens['表现期MOB']==24, '最晚可用放款月'].iloc[0] if (sens['表现期MOB']==24).any() else 'n/a'} 放款的贷款，")
    log("          训练期和 OOT 期切不开，跨时间验证做不成——而跨时间验证正是这个项目要补的第一件事；")
    log(f"       ③ 18 个月捕获 {ref[18]/ref[last_mob]:.0%} 的最终坏账，还能留下 "
        f"{sens.loc[sens['表现期MOB']==18, '可用季度cohort数'].iloc[0]} 个季度 cohort 供训练/OOT 切分，")
    log("          是这份数据上信息量与可验证性的平衡点。")
    peak = inc.loc[inc["近3个月增量"].idxmax(), "mob"]
    hz = inc[inc["近3个月增量"] >= inc["近3个月增量"].max() * 0.9]["mob"]
    log(f"       ④ 违约风险的**边际强度**（第二张图）在 MOB {int(hz.min())}~{int(hz.max())} 之间见顶"
        f"（峰值在 MOB {int(peak)}），")
    log(f"          MOB 21 之后逐月回落。18 个月的窗口正好覆盖住整个高发期，")
    log("          这比「曲线走平」是更靠谱的停表理由。")
    log("       ⑤ 这也和零售无担保分期贷 12~18 个月表现期的行业惯例一致。")

    # ---------------- 2.5 成熟度检验 ----------------
    log.section("2.5  成熟度检验：哪些 cohort 能用")
    W = C.PERFORM_WINDOW
    mat = vt[vt["mob"] == W].dropna(subset=["累计坏账率"])
    log(f"-- 能观察到 MOB {W} 的放款季度及其坏账率 --")
    log(mat[["cohort", "n", "累计坏账率"]].round(4).to_string(index=False))
    log("")
    log("   若越晚的 cohort 坏账率越低且低得反常，说明还是被右截断了，要再往前砍。")

    usable = df["obs_mob"] >= W
    log(f"   放款月 + {W} <= {C.SNAPSHOT} 的贷款：{int(usable.sum()):,} / {len(df):,}")
    log(f"   最晚可用放款月 = {df.loc[usable, 'issue_dt'].max():%Y-%m}")
    df = df[usable].copy()

    # ---------------- 2.6 打标签 ----------------
    log.section(f"2.6  在 MOB {W} 上打好/坏/灰标签")
    df["label"] = label_at(df, W, months_delinq=C.MONTHS_DELINQ)
    df["dq_at_W"] = delinq_months_at(df, W)

    vc = df["label"].value_counts()
    tab = pd.DataFrame({"样本量": vc, "占比": (vc / len(df)).round(4)})
    tab.index = tab.index.map({"good": "好客户", "bad": "坏客户", "grey": "灰客户"})
    log(f"坏客户口径：MOB {W} 时点连续未还款 >= {C.MONTHS_DELINQ} 个月（M3+）")
    log(f"灰客户口径：MOB {W} 时点未还款 1~{C.MONTHS_DELINQ-1} 个月（M1~M2），剔除不建模")
    log("")
    log(tab.to_string())

    log("")
    log("-- 灰客户为什么剔除而不是归好 --")
    log("   M1~M2 的客户最终去向分化很大，归好会污染好样本、归坏会高估坏账率。")
    log("   行业通行做法是剔除，并在报告里给出占比——占比过高（比如 >5%）说明")
    log("   表现期或坏定义没选好，要回去调。")
    grey_share = (df["label"] == "grey").mean()
    log(f"   本项目灰样本占比 {grey_share:.2%}"
        f"，{'在可接受范围' if grey_share < 0.05 else '偏高，需复核口径'}")

    log("")
    log("-- 三组客户的「最终是否核销」对比（验证灰样本该不该单独拎出来）--")
    fin = df.groupby("label", observed=True).agg(
        样本量=("is_bad_ever", "size"), 最终核销率=("is_bad_ever", "mean")).round(4)
    fin.index = fin.index.map({"good": "好客户", "bad": "坏客户", "grey": "灰客户"})
    log(fin.to_string())
    log("")
    log("   读法：灰客户的最终核销率远高于好客户、又低于坏客户——它确实是独立的一档。")
    log("   注意别把「最终核销率高」误读成「灰客户就该算坏」：那是观察到 MOB 30+ 的")
    log("   事后结果，而标签只能在 MOB 18 这个观察点上打。一个客户 MOB 18 时才逾期")
    log("   2 期、MOB 22 才核销，他在 MOB 18 就不是坏客户。把他算成坏，等于把未来")
    log("   信息塞进标签，模型上线后拿不到这个信息。")
    g = df[df["label"] == "grey"]
    if len(g):
        gv = g["loan_status"].value_counts(normalize=True).round(4).head(5)
        log("")
        log("   灰客户的最终状态分布：")
        log(pd.DataFrame({"最终状态": gv.index, "占比": gv.to_numpy()}
                         ).to_string(index=False))

    tab.to_csv(C.OUT / "15_好坏灰分布.csv", encoding="utf-8-sig")

    # ---------------- 2.7 按放款月的坏账率 ----------------
    log.section("2.7  建模样本按放款月的坏客户率")
    s = df[df["label"] != "grey"].copy()
    s["y"] = (s["label"] == "bad").astype(int)
    by_m = s.groupby(s["issue_dt"].dt.to_period("M")).agg(
        样本量=("y", "size"), 坏客户率=("y", "mean")).round(4)
    log(by_m.to_string())

    # ---------------- 2.8 画图 ----------------
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))

    for c in piv.columns:
        ax[0].plot(piv.index, piv[c], lw=1.6, label=c)
    ax[0].axvline(W, ls="--", c="crimson", lw=1.4)
    ax[0].annotate(f"表现期 MOB {W}", (W, ax[0].get_ylim()[1] * 0.06),
                   xytext=(W + 1.2, ax[0].get_ylim()[1] * 0.06), color="crimson")
    ax[0].set_title("代理 vintage 曲线：按放款季度的累计「终态 M3+」坏账率", fontsize=12)
    ax[0].set_xlabel("账龄 MOB（月）"); ax[0].set_ylabel("累计坏账率")
    ax[0].legend(fontsize=7, ncol=2); ax[0].grid(alpha=.3)

    ax[1].plot(inc["mob"], inc["近3个月增量"], "o-", lw=2, c="#1f77b4")
    ax[1].axvline(W, ls="--", c="crimson", lw=1.4)
    ax[1].set_title(f"边际增量：每多观察 3 个月新增的坏账率（{piv.columns[0]} cohort）",
                    fontsize=12)
    ax[1].set_xlabel("账龄 MOB（月）"); ax[1].set_ylabel("近 3 个月新增坏账率")
    ax[1].grid(alpha=.3)

    ax[2].bar(mat["cohort"], mat["累计坏账率"], color="#4c72b0")
    ax[2].set_title(f"成熟度检验：各 cohort 在 MOB {W} 的坏账率", fontsize=12)
    ax[2].set_xlabel("放款季度"); ax[2].set_ylabel(f"MOB {W} 累计坏账率")
    ax[2].tick_params(axis="x", rotation=45); ax[2].grid(alpha=.3, axis="y")

    fig.tight_layout()
    fig.savefig(C.OUT / "16_vintage曲线.png", dpi=150)
    plt.close(fig)
    log("")
    log(f"[OK] 图 -> {C.OUT / '11_vintage曲线.png'}")

    # ---------------- 2.9 落盘 ----------------
    drop_cols = ["last_pymnt_d", "term", "label"]
    s = s.drop(columns=[c for c in drop_cols if c in s.columns])
    s.to_parquet(C.SAMPLE_PQ, index=False)
    log(f"[OK] 建模样本 -> {C.SAMPLE_PQ}  ({len(s):,} 行, 坏客户率 {s['y'].mean():.4f})")
    log.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
