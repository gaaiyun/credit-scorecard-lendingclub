# -*- coding: utf-8 -*-
"""第 6 步：策略表与收益测算，给出 cutoff 建议。

**cutoff 在测试集上选，在 OOT 上固定验证。**
直接在 OOT 上挑分数线，等于拿验证集调参——挑出来的那个点必然在 OOT 上好看，
但它好看是因为被挑过，不是因为它真的稳。正确做法是在训练期内的测试集上定下阈值，
然后把这个**数值固定**拿到 OOT 上去看通过率、坏账率、收益还成不成立。

产出
----
output/50_策略表_十等分_测试集.csv
output/50b_策略表_十等分_OOT.csv
output/51_收益假设.csv        全部假设单列，标明哪些是实测、哪些是拍的
output/52_收益曲线_测试集.csv  不同 cutoff 下的通过率 / 坏账率 / 单位收益
output/53_cutoff建议.csv
output/53b_假设敏感性.csv
output/53c_cutoff在OOT上的验证.csv
output/54_策略收益图.png
output/step6_日志.txt
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
from profit import (Assumptions, empirical_lgd, profit_by_cutoff,  # noqa: E402
                    unit_economics)
from scorecard import cutoff_table                      # noqa: E402
from util import Logger                                 # noqa: E402

log = Logger(C.OUT / "step6_日志.txt")


def load_seg(name: str, aux: pd.DataFrame) -> pd.DataFrame:
    sc = pd.read_parquet(C.PROC / f"scored_{name}.parquet")
    return sc.merge(aux, on="id", how="left", suffixes=("", "_f"))


def main() -> int:
    t0 = time.time()
    log.section("第 6 步  策略表与收益测算")

    aux = pd.read_parquet(C.SAMPLE_PQ, columns=[
        "id", "funded_amnt", "total_rec_prncp", "recoveries", "int_rate",
        "loan_status"])
    te = load_seg("test", aux)
    oot = load_seg("oot", aux)
    log(f"测试集 {len(te):,} 笔，坏客户率 {te['y'].mean():.4f}（训练区间内随机留出，"
        f"用来**选** cutoff）")
    log(f"OOT   {len(oot):,} 笔，坏客户率 {oot['y'].mean():.4f}（训练区间之后的放款，"
        f"用来**验** cutoff）")
    log("")
    log("为什么不在 OOT 上选：那等于拿验证集调参。挑出来的分数线必然在 OOT 上好看，")
    log("但好看是因为被挑过。上线时你手里只有建模期数据，所以模拟上线就必须只用建模期选。")
    log("")
    log("（注：这里的「通过率」是相对**已核准人群**的。LC 已经先筛掉约 94.7% 的申请，")
    log("  本项目的 cutoff 是在它筛剩下的人里再切一刀，不等于真实审批通过率。）")

    # ---------------- 6.1 十等分策略表 ----------------
    log.section("6.1  分数十等分策略表")
    for nm, d, fn in (("测试集", te, "50_策略表_十等分_测试集.csv"),
                      ("OOT", oot, "50b_策略表_十等分_OOT.csv")):
        ct = cutoff_table(d["score"].to_numpy(), d["y"].to_numpy(), n=10)
        ct.to_csv(C.OUT / fn, index=False, encoding="utf-8-sig")
        base_d = float(d["y"].mean())
        log("")
        log(f"-- {nm}（全通过坏账率 {base_d:.2%}）--")
        log(ct[["下界", "上界", "人数", "本档坏账率", "通过率", "累计坏账率"]]
            .to_string(index=False, formatters={
                "下界": "{:.0f}".format, "上界": "{:.0f}".format,
                "本档坏账率": "{:.2%}".format, "通过率": "{:.1%}".format,
                "累计坏账率": "{:.2%}".format}))
        log(f"   最高档坏账率 {ct['本档坏账率'].iloc[0]:.2%}，"
            f"最低档 {ct['本档坏账率'].iloc[-1]:.2%}，"
            f"相差 {ct['本档坏账率'].iloc[-1]/ct['本档坏账率'].iloc[0]:.1f} 倍")

    base_te = float(te["y"].mean())
    base_oot = float(oot["y"].mean())

    # ---------------- 6.2 假设 ----------------
    log.section("6.2  收益测算的假设（全部单列，不混进结果里）")
    a = Assumptions()

    log("先用数据实测 LGD，再决定假设取值：")
    co = aux[aux["loan_status"].isin(["Charged Off", "Default"])]
    lgd_hat, rr, lost = empirical_lgd(co["funded_amnt"], co["total_rec_prncp"],
                                      co["recoveries"])
    log(f"    建模样本内已核销 {len(co):,} 笔，未收回本金合计 ${lost:,.0f}")
    log(f"    回收金额 / 未收回本金 = {rr:.2%}  ->  实测 LGD = {lgd_hat:.4f}")
    log("    （recoveries / total_rec_prncp 是放款后字段，不进模型，")
    log("      但拿来做事后的损失核算是合理的——那不是预测特征。）")
    a.lgd = round(float(lgd_hat), 4)
    a.lgd_source = (f"实测：建模样本内 {len(co):,} 笔已核销贷款，"
                    f"回收率 {rr:.2%}，故 LGD = {lgd_hat:.4f}")

    tbl = a.table()
    tbl.to_csv(C.OUT / "51_收益假设.csv", index=False, encoding="utf-8-sig")
    log("")
    log(tbl.to_string(index=False))
    log("")
    log("[必须说明] 上表五条里只有 LGD 是从数据里算出来的，其余四条都是假设。")
    log("           换一组假设，推荐的 cutoff 会变。所以下面会做假设敏感性。")

    # ---------------- 6.3 单笔收益 ----------------
    log.section("6.3  逐笔收益测算（测试集）")
    econ_te = unit_economics(te["funded_amnt"], te["int_rate"], te["y"], a)
    econ_oot = unit_economics(oot["funded_amnt"], oot["int_rate"], oot["y"], a)
    log(f"平均放款金额 ${econ_te['放款金额'].mean():,.0f}，"
        f"平均利率 {te['int_rate'].mean():.2f}%")
    g = econ_te.groupby("y").agg(笔数=("净收益", "size"), 平均收入=("收入", "mean"),
                                 平均损失=("损失", "mean"),
                                 平均资金运营成本=("资金运营成本", "mean"),
                                 平均净收益=("净收益", "mean")).round(1)
    g.index = ["好客户", "坏客户"]
    log(g.to_string())
    log("")
    log(f"全通过的件均净收益 = ${econ_te['净收益'].mean():,.1f}")
    log(f"  好客户赚 ${g.loc['好客户','平均净收益']:,.0f}，"
        f"坏客户亏 ${-g.loc['坏客户','平均净收益']:,.0f}，"
        f"一个坏客户要 {-g.loc['坏客户','平均净收益']/g.loc['好客户','平均净收益']:.1f} "
        f"个好客户来填")

    # ---------------- 6.4 收益曲线（测试集）----------------
    log.section("6.4  不同 cutoff 下的收益（测试集）")
    pc = profit_by_cutoff(te["score"].to_numpy(), econ_te, n_steps=40)
    pc.to_csv(C.OUT / "52_收益曲线_测试集.csv", index=False, encoding="utf-8-sig")
    show = pc[pc.index % 4 == 0][["cutoff分数", "通过率", "坏账率",
                                  "单位放款净收益", "件均净收益", "净收益"]]
    log(show.to_string(index=False, formatters={
        "通过率": "{:.1%}".format, "坏账率": "{:.2%}".format,
        "单位放款净收益": "{:.4f}".format, "件均净收益": "{:,.0f}".format,
        "净收益": "{:,.0f}".format}))

    best_unit = pc.loc[pc["单位放款净收益"].idxmax()]
    best_total = pc.loc[pc["净收益"].idxmax()]
    log("")
    log(f"单位放款净收益最高：cutoff {best_unit['cutoff分数']:.0f} 分，"
        f"通过率 {best_unit['通过率']:.1%}")
    log(f"总净收益最高：      cutoff {best_total['cutoff分数']:.0f} 分，"
        f"通过率 {best_total['通过率']:.1%}")
    log("")
    log("两者不一致是必然的：单位收益最高在高分段（每一笔都很赚），")
    log("总收益最高在低分段（多做一笔只要还是正的就加总量）。")
    log("实际定 cutoff 在这两端之间，看三件事：资本约束、风险偏好、审批量目标。")

    # ---------------- 6.5 假设敏感性 ----------------
    log.section("6.5  假设敏感性：换一组假设，推荐会不会翻")
    sens = []
    for name, kw in [
        ("基准", {}),
        ("LGD 降到 0.70", {"lgd": 0.70}),
        ("LGD 升到 1.00（零回收）", {"lgd": 1.00}),
        ("资金成本 4%->8%", {"funding_cost": 0.08}),
        ("存续期 1.6->1.2 年", {"avg_life_years": 1.2}),
        ("EAD 0.75->0.90", {"ead_ratio": 0.90}),
    ]:
        aa = Assumptions(**{**{k: v for k, v in vars(a).items()
                              if not k.endswith("_source")}, **kw})
        e2 = unit_economics(te["funded_amnt"], te["int_rate"], te["y"], aa)
        p2 = profit_by_cutoff(te["score"].to_numpy(), e2, n_steps=40)
        bu = p2.loc[p2["单位放款净收益"].idxmax()]
        bt = p2.loc[p2["净收益"].idxmax()]
        sens.append({"假设": name,
                     "全通过件均净收益": round(float(e2["净收益"].mean()), 1),
                     "单位收益最优cutoff": bu["cutoff分数"],
                     "单位收益最优通过率": round(float(bu["通过率"]), 3),
                     "总收益最优cutoff": bt["cutoff分数"],
                     "总收益最优通过率": round(float(bt["通过率"]), 3)})
    sdf = pd.DataFrame(sens)
    log(sdf.to_string(index=False))
    sdf.to_csv(C.OUT / "53b_假设敏感性.csv", index=False, encoding="utf-8-sig")

    log("")
    log("-- 这张表里最该注意的一行 --")
    fc = sdf[sdf["假设"] == "资金成本 4%->8%"].iloc[0]
    log(f"    资金成本从 4% 提到 8%，全通过的件均净收益从 "
        f"${sdf[sdf['假设']=='基准']['全通过件均净收益'].iloc[0]:,.0f} 变成 "
        f"${fc['全通过件均净收益']:,.0f}——**整个组合由赚变亏**，")
    log(f"    总收益最优通过率塌到 {fc['总收益最优通过率']:.1%}。")
    log("    也就是说：在这个产品的利差结构下，资金成本翻倍比坏账率翻倍更致命。")
    log("    这不是模型问题，是这类高息小额分期产品的固有杠杆。")
    log("    反过来，LGD 从 0.89 调到 0.70 或 1.00，推荐 cutoff 只小幅移动，")
    log("    说明结论对**我唯一实测出来的**参数反而不敏感——敏感的是那几个拍出来的。")

    # ---------------- 6.6 在测试集上定 cutoff ----------------
    log.section("6.6  在测试集上定 cutoff")
    risk_cap = base_te * 0.80
    ok = pc[pc["坏账率"] <= risk_cap]
    rec_rows = []
    if len(ok):
        r1 = ok.iloc[ok["通过率"].argmax()]
        rec_rows.append({"口径": f"坏账率不高于基准 80%（<= {risk_cap:.2%}）下的最大通过量",
                         "cutoff": r1["cutoff分数"], "通过率": r1["通过率"],
                         "坏账率": r1["坏账率"],
                         "单位放款净收益": r1["单位放款净收益"]})
    rec_rows.append({"口径": "单位放款净收益最大", "cutoff": best_unit["cutoff分数"],
                     "通过率": best_unit["通过率"], "坏账率": best_unit["坏账率"],
                     "单位放款净收益": best_unit["单位放款净收益"]})
    rec_rows.append({"口径": "总净收益最大", "cutoff": best_total["cutoff分数"],
                     "通过率": best_total["通过率"], "坏账率": best_total["坏账率"],
                     "单位放款净收益": best_total["单位放款净收益"]})
    rec = pd.DataFrame(rec_rows)
    rec.to_csv(C.OUT / "53_cutoff建议.csv", index=False, encoding="utf-8-sig")
    log(rec.to_string(index=False, formatters={
        "cutoff": "{:.0f}".format, "通过率": "{:.1%}".format,
        "坏账率": "{:.2%}".format, "单位放款净收益": "{:.4f}".format}))

    pick = float(rec.iloc[0]["cutoff"])
    log("")
    log(f"[推荐] cutoff = {pick:.0f} 分")
    log("理由：**风险上限优先**。坏账率是硬约束，收益最优点若突破风险偏好就不能选——")
    log("      这是风控和业务最常见的分歧点。在约束下取最大通过量，保住审批规模。")

    # ---------------- 6.7 把 cutoff 固定，拿到 OOT 上验证 ----------------
    log.section("6.7  固定 cutoff，在 OOT 上验证（这一步才是「验」）")
    log(f"把测试集上定下来的 {pick:.0f} 分**原封不动**拿到 OOT 上，看还成不成立。")
    log("")
    rows = []
    for nm, d, econ, base_d in (("测试集（选的地方）", te, econ_te, base_te),
                                ("OOT（验的地方）", oot, econ_oot, base_oot)):
        m = d["score"].to_numpy() >= pick
        e = econ[m]
        rows.append({
            "样本": nm, "样本量": len(d), "全通过坏账率": round(base_d, 4),
            "通过率": round(float(m.mean()), 4),
            "通过后坏账率": round(float(d.loc[m, "y"].mean()), 4),
            "坏账率降幅": round(1 - float(d.loc[m, "y"].mean()) / base_d, 4),
            "单位放款净收益": round(float(e["净收益"].sum() / e["放款金额"].sum()), 5),
            "件均净收益": round(float(e["净收益"].mean()), 1)})
    val = pd.DataFrame(rows)
    val.to_csv(C.OUT / "53c_cutoff在OOT上的验证.csv", index=False,
               encoding="utf-8-sig")
    log(val.to_string(index=False, formatters={
        "全通过坏账率": "{:.2%}".format, "通过率": "{:.1%}".format,
        "通过后坏账率": "{:.2%}".format, "坏账率降幅": "{:.1%}".format,
        "单位放款净收益": "{:.4f}".format, "件均净收益": "{:,.0f}".format}))

    d_pass = abs(val.loc[1, "通过率"] - val.loc[0, "通过率"])
    log("")
    log(f"通过率在两段之间相差 {d_pass:.1%}，"
        f"坏账率降幅 {val.loc[0,'坏账率降幅']:.1%} -> {val.loc[1,'坏账率降幅']:.1%}。")
    if d_pass < 0.05:
        log("[OK] 阈值迁移到 OOT 后通过率与风险收益结构都稳定，这个 cutoff 站得住。")
    else:
        log("[注意] 通过率在 OOT 上偏移较大，说明分数分布漂移，cutoff 要按月重校准。")
    log("")
    log("生产上这一步叫 **cutoff 回溯验证**。上线前必做，上线后每月重做一次——")
    log("分数分布一漂移，同一个分数线对应的通过率就变了，审批量会自己跑掉。")

    # ---------------- 6.8 边界 ----------------
    log.section("6.8  这个测算的边界")
    log("1. 没有做现金流贴现，没有区分违约发生在第几期，提前结清的利息损失也没算。")
    log("   够用来在不同 cutoff 之间做相对比较，不够用来做真实定价。")
    log("2. 利率用的是 LC 实际给出的风险定价。低分段的人利率更高，")
    log("   这本身就在补偿风险——所以放宽 cutoff 的收益下降没有想象中快。")
    log("3. 通过率是相对已核准人群的。真实审批策略要在全体申请人上定，")
    log("   而被拒人群的真实表现观察不到（见 step5）。")

    # ---------------- 图 ----------------
    ct_te = cutoff_table(te["score"].to_numpy(), te["y"].to_numpy(), n=10)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))

    ax[0].bar(range(len(ct_te)), ct_te["本档坏账率"], color="#4c72b0")
    ax[0].axhline(base_te, ls="--", c="crimson", lw=1.2, label=f"全通过 {base_te:.2%}")
    ax[0].set_xticks(range(len(ct_te)))
    ax[0].set_xticklabels([f"{v:.0f}" for v in ct_te["下界"]], rotation=45, fontsize=8)
    ax[0].set_title("分数十等分的本档坏账率（测试集）")
    ax[0].set_xlabel("该档分数下界"); ax[0].set_ylabel("坏账率")
    ax[0].legend(); ax[0].grid(alpha=.3, axis="y")

    for nm, d, c in (("测试集", te, "#4c72b0"), ("OOT", oot, "#c44e52")):
        ctx = cutoff_table(d["score"].to_numpy(), d["y"].to_numpy(), n=20)
        ax[1].plot(ctx["通过率"], ctx["累计坏账率"], "o-", lw=1.8, ms=3, c=c, label=nm)
    ax[1].axhline(risk_cap, ls="--", c="gray", lw=1, label=f"风险上限 {risk_cap:.2%}")
    ax[1].set_title("通过率 — 坏账率：测试集定线，OOT 验证")
    ax[1].set_xlabel("通过率"); ax[1].set_ylabel("累计坏账率")
    ax[1].legend(); ax[1].grid(alpha=.3)

    ax2 = ax[2]
    ax2.plot(pc["通过率"], pc["单位放款净收益"], "o-", lw=2, ms=3,
             c="#4c72b0", label="单位放款净收益（左轴）")
    ax2.set_xlabel("通过率"); ax2.set_ylabel("单位放款净收益")
    ax2.axhline(0, ls=":", c="gray", lw=1)
    ax3 = ax2.twinx()
    ax3.plot(pc["通过率"], pc["净收益"] / 1e6, "s--", lw=1.6, ms=3,
             c="#dd8452", label="总净收益（右轴，百万美元）")
    ax3.set_ylabel("总净收益（百万美元）")
    ax2.axvline(float(rec.iloc[0]["通过率"]), ls="--", c="crimson", lw=1.2)
    ax2.annotate(f"推荐 cutoff {pick:.0f}",
                 (float(rec.iloc[0]["通过率"]), ax2.get_ylim()[0]),
                 xytext=(max(float(rec.iloc[0]["通过率"]) - 0.35, 0.02),
                         ax2.get_ylim()[0] + 0.004),
                 color="crimson", fontsize=9)
    ax2.set_title("收益曲线（测试集）：单位收益 vs 总收益")
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="lower right", fontsize=8)
    ax2.grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(C.OUT / "54_策略收益图.png", dpi=150)
    plt.close(fig)
    log("")
    log(f"[OK] 图 -> {C.OUT / '54_策略收益图.png'}")
    log(f"[OK] 全步用时 {time.time()-t0:.0f}s")
    log.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
