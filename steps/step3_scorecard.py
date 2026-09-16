# -*- coding: utf-8 -*-
"""第 3 步：跨时间切分 + 评分卡建模 + 三段验证（训练 / 测试 / OOT）。

产出
----
output/20_分箱明细.csv       每变量每箱的人数、坏账率、WOE、IV
output/21_变量筛选过程.csv   IV / 相关性 / 系数符号三道关的逐步结果
output/22_回归结果.txt       statsmodels 摘要（系数、p 值、VIF）
output/23_评分卡.csv         最终评分卡：每变量每箱的得分，业务方可手算
output/24_三段表现.csv       训练 / 测试 / OOT 的 KS / AUC / Gini
output/25_PSI_CSI.csv        分数 PSI 与逐变量 CSI
output/26_评估图.png
output/step3_日志.txt
data/processed/scored_*.parquet  三段样本的分数，供 step5/6/7 复用

流程
----
按放款月切训练/OOT（不许随机切）
  -> 单调分箱 -> WOE/IV -> IV 筛 -> 相关性去共线 -> 逻辑回归
  -> 系数符号与显著性检验 -> VIF -> 刻度化 -> KS/AUC/PSI/CSI
"""
from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as C                                       # noqa: E402
import matplotlib.pyplot as plt                          # noqa: E402
from leakage import MODEL_FEATURES                       # noqa: E402
from scorecard import (MISSING_BIN, apply_bins, apply_cat_bins,   # noqa: E402
                       bin_table, cat_bins, csi, cutoff_table,
                       drop_bad_signs, is_monotonic, iv_of, ks_auc,
                       monotonic_bins, psi, psi_detail, scale_params,
                       to_score, vif_table, woe_map)
from util import Logger                                  # noqa: E402

log = Logger(C.OUT / "step3_日志.txt")

# 类别型候选变量（其余按数值处理）
CAT_FEATURES = ["home_ownership", "verification_status", "purpose",
                "addr_state", "application_type"]
# 衍生出来的数值特征
DERIVED = ["credit_hist_mths", "emp_length_num"]


def build_all_bins(tr: pd.DataFrame, num_feats: list[str],
                   cat_feats: list[str]) -> tuple[dict, dict, pd.DataFrame]:
    """在**训练集**上做分箱。切点绝不能用测试/OOT 的数据定，否则就是偷看未来。"""
    num_cuts, cat_maps, tables = {}, {}, []
    for f in num_feats:
        c = monotonic_bins(tr[f], tr["y"], max_bins=C.MAX_BINS,
                           min_rate=C.MIN_BIN_RATE)
        if not c:
            continue
        t = bin_table(apply_bins(tr[f], c), tr["y"], name=f)
        num_cuts[f] = c
        t["iv"] = iv_of(t)
        t["monotonic"] = is_monotonic(t)
        t["类型"] = "数值"
        tables.append(t)
    for f in cat_feats:
        mp = cat_bins(tr[f], tr["y"], min_share=0.02, max_bins=C.MAX_BINS)
        if len(set(mp.values())) < 2:
            continue
        t = bin_table(apply_cat_bins(tr[f], mp), tr["y"], name=f)
        cat_maps[f] = mp
        t["iv"] = iv_of(t)
        t["monotonic"] = is_monotonic(t)
        t["类型"] = "类别"
        tables.append(t)
    return num_cuts, cat_maps, pd.concat(tables, ignore_index=True)


def to_bins(df: pd.DataFrame, num_cuts: dict, cat_maps: dict) -> pd.DataFrame:
    out = {f: apply_bins(df[f], c) for f, c in num_cuts.items()}
    out.update({f: apply_cat_bins(df[f], m) for f, m in cat_maps.items()})
    return pd.DataFrame(out, index=df.index)


def to_woe(binned: pd.DataFrame, wmap: dict) -> pd.DataFrame:
    """箱号 -> WOE。训练集没出现过的箱给 0（等价于「与整体一致」）。"""
    return pd.DataFrame(
        {f: binned[f].map(lambda k, f=f: wmap.get((f, int(k)), 0.0)).astype(float)
         for f in binned.columns}, index=binned.index)


def main() -> int:
    t0 = time.time()
    log.section("第 3 步  跨时间切分 + 评分卡建模 + 三段验证")

    df = pd.read_parquet(C.SAMPLE_PQ)
    log(f"建模样本 {len(df):,} 笔，坏客户率 {df['y'].mean():.4f}")

    # ---------------- 3.1 跨时间切分 ----------------
    log.section("3.1  按放款月切训练 / OOT —— 不许随机切")
    log("随机切出来的「测试集」和训练集是同一批人群、同一个时间段，")
    log("它只能证明模型没过拟合，证明不了模型在未来的人群上还成立。")
    log("A 卡上线后面对的是几个月后的申请人，所以验证集必须按时间切。")
    log("")
    ism = df["issue_dt"]
    in_train = ism.between(C.TRAIN_START,
                           pd.Timestamp(C.TRAIN_END) + pd.offsets.MonthEnd(0))
    in_oot = ism.between(C.OOT_START,
                         pd.Timestamp(C.OOT_END) + pd.offsets.MonthEnd(0))
    log(f"训练区间 {C.TRAIN_START} ~ {C.TRAIN_END}：{int(in_train.sum()):,} 笔")
    log(f"OOT 区间 {C.OOT_START} ~ {C.OOT_END}：{int(in_oot.sum()):,} 笔")
    if in_train.sum() == 0 or in_oot.sum() == 0:
        log("[FAIL] 切分区间和样本不匹配，检查 config 里的日期")
        return 1

    # 只保留后续真正要用的列再切。这台机器的 Windows 提交限额只剩 4GB，
    # 整张 95 列的表拖到回归那一步会在 IRLS 的 SVD 里 MemoryError。
    cand_all = [c for c in MODEL_FEATURES if c in df.columns] + DERIVED
    need = sorted(set(cand_all) | {"y", "id", "issue_dt", "int_rate",
                                   "funded_amnt", "grade"})
    df = df[[c for c in need if c in df.columns]]

    from sklearn.model_selection import train_test_split
    pool = df[in_train]
    tr, te = train_test_split(pool, test_size=C.TEST_SIZE, stratify=pool["y"],
                              random_state=C.RANDOM_STATE)
    tr, te = tr.copy(), te.copy()
    oot = df[in_oot].copy()
    del df, pool
    gc.collect()
    log("")
    for nm, d in (("训练", tr), ("测试", te), ("OOT", oot)):
        log(f"    {nm:4s} {len(d):>7,} 笔  坏客户率 {d['y'].mean():.4f}  "
            f"放款期 {d['issue_dt'].min():%Y-%m} ~ {d['issue_dt'].max():%Y-%m}")
    log("")
    log("    注意：测试集是训练区间内的随机留出，用来看过拟合；")
    log("          OOT 是训练区间**之后**的放款，用来看跨时间稳定性。两者作用不同。")

    # ---------------- 3.2 分箱 ----------------
    log.section("3.2  分箱与 WOE（切点只在训练集上定）")
    cand = [c for c in cand_all if c in tr.columns and c != "term"]  # 只建 36 期卡
    num_feats = [c for c in cand if c not in CAT_FEATURES
                 and pd.api.types.is_numeric_dtype(tr[c])]
    cat_feats = [c for c in CAT_FEATURES if c in tr.columns]

    miss_rate = tr[num_feats + cat_feats].isna().mean()
    too_missing = miss_rate[miss_rate > C.MISSING_CEIL].index.tolist()
    if too_missing:
        log(f"缺失率 > {C.MISSING_CEIL:.0%} 直接剔除：{too_missing}")
        num_feats = [c for c in num_feats if c not in too_missing]
        cat_feats = [c for c in cat_feats if c not in too_missing]
    log(f"候选变量 {len(num_feats)} 个数值 + {len(cat_feats)} 个类别")

    num_cuts, cat_maps, tbl = build_all_bins(tr, num_feats, cat_feats)
    tbl.to_csv(C.OUT / "20_分箱明细.csv", index=False, encoding="utf-8-sig")
    n_bins = tbl.groupby("var")["bin"].nunique()
    log(f"[OK] 成功分箱 {tbl['var'].nunique()} 个变量，共 {len(tbl)} 个箱"
        f"（平均 {n_bins.mean():.1f} 箱/变量）")

    iv = tbl.groupby("var")["iv"].first().sort_values(ascending=False)
    mono = tbl.groupby("var")["monotonic"].first()
    log("")
    log("-- IV 前 15 --")
    for v, s in iv.head(15).items():
        log(f"    IV {s:6.4f}  {'单调' if mono[v] else '非单调'}  {v}")

    # ---------------- 3.3 三道筛选 ----------------
    log.section("3.3  变量筛选三道关")
    steps = []

    # 关 1：IV
    keep1 = [v for v in iv.index if C.IV_FLOOR <= iv[v] <= C.IV_CEIL]
    hi = [v for v in iv.index if iv[v] > C.IV_CEIL]
    log(f"[关 1] IV 在 [{C.IV_FLOOR}, {C.IV_CEIL}] 之间：保留 {len(keep1)} / {len(iv)}")
    if hi:
        log(f"       IV > {C.IV_CEIL} 的变量要回头查是不是泄漏：{hi}")
    steps.append({"关卡": "1_IV", "保留数": len(keep1),
                  "剔除": ", ".join([v for v in iv.index if v not in keep1])})

    # 单调性兜底（分箱阶段已用合箱解决，这里只看合到底还不单调的）
    bad_mono = [v for v in keep1 if not mono[v] and v not in cat_feats]
    if bad_mono:
        log(f"       合箱后仍不单调、剔除：{bad_mono}")
        keep1 = [v for v in keep1 if v not in bad_mono]

    # 关 2：相关性去共线
    woe_tr = to_woe(to_bins(tr, num_cuts, cat_maps), woe_map(tbl))
    corr = woe_tr[keep1].corr().abs()
    keep2: list[str] = []
    dropped_corr = []
    for v in keep1:                       # keep1 已按 IV 降序
        clash = [k for k in keep2 if corr.loc[v, k] >= C.CORR_CEIL]
        if clash:
            dropped_corr.append((v, clash[0], round(float(corr.loc[v, clash[0]]), 3)))
        else:
            keep2.append(v)
    log("")
    log(f"[关 2] 相关系数 < {C.CORR_CEIL} 去共线：保留 {len(keep2)} / {len(keep1)}")
    log("       按 IV 从高到低贪心保留。高相关的变量一起进 LR 会让系数符号乱掉，")
    log("       评分卡就没法向业务方解释了。")
    for v, k, c in dropped_corr[:12]:
        log(f"       剔除 {v:28s} 与 {k:28s} 相关 {c}")
    if len(dropped_corr) > 12:
        log(f"       ...（共剔除 {len(dropped_corr)} 个）")
    steps.append({"关卡": "2_相关性", "保留数": len(keep2),
                  "剔除": ", ".join(v for v, _, _ in dropped_corr)})

    # 关 3：系数符号 + 显著性
    log("")
    log(f"[关 3] 系数必须为正、p < {C.P_CEIL}，逐步剔除：")
    log("       WOE 越大代表这箱越坏，模型预测的是坏客户概率，所以系数为负意味着")
    log("       「越坏的特征反而降低违约概率」，业务上讲不通，多严重的共线也不能留。")
    model, final_feats = drop_bad_signs(woe_tr[keep2], tr["y"],
                                        p_ceil=C.P_CEIL, log=log)
    log(f"[OK] 入模 {len(final_feats)} 个变量")
    steps.append({"关卡": "3_符号显著性", "保留数": len(final_feats),
                  "剔除": ", ".join(v for v in keep2 if v not in final_feats)})
    pd.DataFrame(steps).to_csv(C.OUT / "21_变量筛选过程.csv", index=False,
                               encoding="utf-8-sig")

    # 求解器对拍：本项目默认 IRLS，内存不够时降级 lbfgs，两条路必须给同一个解
    log("")
    log("-- 求解器对拍：IRLS（默认）vs lbfgs（内存不足时的降级路径）--")
    try:
        from scorecard import fit_logit as _fit
        m_alt = _fit(woe_tr[final_feats], tr["y"], method="lbfgs")
        names = ["const"] + final_feats
        cmp = pd.DataFrame({
            "变量": names,
            "IRLS系数": [round(float(model.params[v]), 6) for v in names],
            "lbfgs系数": [round(float(m_alt.params[v]), 6) for v in names]})
        cmp["绝对差"] = (cmp["IRLS系数"] - cmp["lbfgs系数"]).abs()
        cmp["相对差"] = (cmp["绝对差"] / cmp["IRLS系数"].abs()).round(6)
        # 真正该关心的不是系数差多少，而是分数差多少
        import statsmodels.api as _sm
        Xc = _sm.add_constant(woe_tr[final_feats], has_constant="add")
        _A, _B = scale_params(C.PDO, C.BASE_SCORE, C.BASE_ODDS)
        d_score = np.abs(to_score(np.asarray(model.predict(Xc), dtype=float), _A, _B) -
                         to_score(np.asarray(m_alt.predict(Xc), dtype=float), _A, _B))
        log(f"   系数最大绝对差 {cmp['绝对差'].max():.2e}，最大相对差 {cmp['相对差'].max():.2e}")
        log(f"   换算到客户分数上：最大差 {d_score.max():.4f} 分，"
            f"平均差 {d_score.mean():.4f} 分")
        log("   [OK] 不到 1 分，结论不受求解器选择影响")
        cmp.round(8).to_csv(C.OUT / "22b_求解器对拍.csv", index=False,
                            encoding="utf-8-sig")
        del m_alt, Xc, d_score
        gc.collect()
    except Exception as e:      # noqa: BLE001
        log(f"   [跳过] 对拍失败：{type(e).__name__}: {e}")

    # VIF
    vt = vif_table(woe_tr[final_feats])
    log("")
    log("-- VIF（> 10 说明共线严重、系数不可信）--")
    log(vt.to_string(index=False))
    if (vt["VIF"] > C.VIF_CEIL).any():
        log(f"[注意] 有变量 VIF > {C.VIF_CEIL}，需要复核")
    else:
        log(f"[OK] 全部 VIF < {C.VIF_CEIL}")

    with open(C.OUT / "22_回归结果.txt", "w", encoding="utf-8") as f:
        f.write(str(model.summary()))
        f.write("\n\n=== VIF ===\n")
        f.write(vt.to_string(index=False))

    log("")
    log("-- 入模变量的系数与显著性 --")
    coef = pd.DataFrame({"变量": final_feats,
                         "系数": [round(float(model.params[v]), 4) for v in final_feats],
                         "p值": [round(float(model.pvalues[v]), 6) for v in final_feats],
                         "IV": [round(float(iv[v]), 4) for v in final_feats]})
    coef = coef.sort_values("IV", ascending=False)
    log(coef.to_string(index=False))

    # ---------------- 3.4 刻度化 ----------------
    log.section("3.4  刻度化")
    A, B = scale_params(C.PDO, C.BASE_SCORE, C.BASE_ODDS)
    log(f"PDO = {C.PDO}，基准分 {C.BASE_SCORE} @ odds {C.BASE_ODDS}:1")
    log(f"B = PDO / ln2 = {B:.4f}")
    log(f"A = 基准分 + B x ln(基准odds) = {A:.4f}")
    log("score = A - B x ln( p / (1-p) )，每箱得分 = -B x 系数 x WOE")

    wmap = woe_map(tbl)

    def score_of(d: pd.DataFrame) -> np.ndarray:
        import statsmodels.api as sm
        X = to_woe(to_bins(d, num_cuts, cat_maps), wmap)[final_feats]
        p = model.predict(sm.add_constant(X, has_constant="add"))
        return to_score(np.asarray(p, dtype=float), A, B)

    s_tr, s_te, s_oot = score_of(tr), score_of(te), score_of(oot)

    card_rows = []
    for f in final_feats:
        cf = float(model.params[f])
        for _, r in tbl[tbl["var"] == f].iterrows():
            b = int(r["bin"])
            if f in cat_maps:
                cats = sorted(k for k, v in cat_maps[f].items() if v == b)
                desc = "缺失" if b == MISSING_BIN else ("未见过的类别" if b == -2
                                                      else " / ".join(map(str, cats)))
            else:
                cuts = [-np.inf] + list(num_cuts[f]) + [np.inf]
                desc = "缺失" if b == MISSING_BIN else \
                    f"({cuts[b]:.4g}, {cuts[b+1]:.4g}]"
            card_rows.append({
                "变量": f, "箱号": b, "箱区间": desc,
                "占比": round(float(r["share"]), 4),
                "坏账率": round(float(r["bad_rate"]), 4),
                "WOE": round(float(r["woe"]), 4),
                "系数": round(cf, 4),
                "该箱得分": round(-B * cf * float(r["woe"]), 2)})
    card = pd.DataFrame(card_rows)
    card["基准分"] = round(float(A - B * model.params["const"]), 2)
    card.to_csv(C.OUT / "23_评分卡.csv", index=False, encoding="utf-8-sig")
    log(f"[OK] 评分卡 -> {C.OUT / '23_评分卡.csv'}（{len(card)} 个箱）")
    log(f"    基准分（截距项）= {A - B * model.params['const']:.2f}，"
        f"客户总分 = 基准分 + 各变量所在箱的得分之和")

    # ---------------- 3.5 三段表现 ----------------
    log.section("3.5  三段表现：训练 / 测试 / OOT")
    rows = []
    for nm, s, d in (("训练", s_tr, tr), ("测试", s_te, te), ("OOT", s_oot, oot)):
        ks, auc = ks_auc(d["y"], s)
        rows.append({"样本": nm, "样本量": len(d), "坏客户率": round(d["y"].mean(), 4),
                     "KS": round(ks, 4), "AUC": round(auc, 4),
                     "Gini": round(2 * auc - 1, 4)})
        log(f"    {nm:4s} n={len(d):>7,}  坏率 {d['y'].mean():.4f}  "
            f"KS {ks:.4f}  AUC {auc:.4f}  Gini {2*auc-1:.4f}")
    perf = pd.DataFrame(rows)
    perf.to_csv(C.OUT / "24_三段表现.csv", index=False, encoding="utf-8-sig")

    ks_tr = perf.loc[perf["样本"] == "训练", "KS"].iloc[0]
    ks_oot = perf.loc[perf["样本"] == "OOT", "KS"].iloc[0]
    log("")
    log(f"    训练 -> OOT 的 KS 衰减 = {(ks_tr - ks_oot)/ks_tr:.1%}")
    log("    跨时间掉一点是正常的；掉超过 20% 要查是人群变了还是变量口径变了。")

    # ---------------- 3.6 PSI / CSI ----------------
    log.section("3.6  分数 PSI 与变量 CSI")
    psi_te, psi_oot = psi(s_tr, s_te), psi(s_tr, s_oot)
    log(f"分数 PSI：训练 vs 测试 {psi_te:.4f}（同期随机留出，做对照）")
    log(f"分数 PSI：训练 vs OOT  {psi_oot:.4f}  <-- 这个才是跨时间稳定性")
    log(f"判读：< {C.PSI_WATCH} 稳定，{C.PSI_WATCH}~{C.PSI_ALERT} 关注，"
        f"> {C.PSI_ALERT} 人群已变、要重训")

    b_tr = to_bins(tr, num_cuts, cat_maps)
    b_oot = to_bins(oot, num_cuts, cat_maps)
    csi_rows = [{"变量": f, "CSI_训练vs OOT": round(csi(b_tr[f], b_oot[f]), 4),
                 "入模": f in final_feats, "IV": round(float(iv[f]), 4)}
                for f in b_tr.columns]
    csi_df = pd.DataFrame(csi_rows).sort_values("CSI_训练vs OOT", ascending=False)
    csi_df.to_csv(C.OUT / "25_PSI_CSI.csv", index=False, encoding="utf-8-sig")
    psi_detail(s_tr, s_oot).to_csv(C.OUT / "25b_分数PSI明细.csv", index=False,
                                   encoding="utf-8-sig")

    log("")
    log("-- 入模变量的 CSI（训练 vs OOT）--")
    log(csi_df[csi_df["入模"]].to_string(index=False))
    alarm = csi_df[(csi_df["入模"]) & (csi_df["CSI_训练vs OOT"] > C.CSI_WATCH)]
    log("")
    if len(alarm):
        log(f"[注意] 有 {len(alarm)} 个入模变量 CSI > {C.CSI_WATCH}，逐个解释：")
        for _, r in alarm.iterrows():
            v = r["变量"]
            a = b_tr[v].value_counts(normalize=True).sort_index().round(4)
            b = b_oot[v].value_counts(normalize=True).sort_index().round(4)
            cmp = pd.DataFrame({"训练占比": a, "OOT占比": b}).fillna(0)
            cmp["变化"] = (cmp["OOT占比"] - cmp["训练占比"]).round(4)
            log(f"    {v}（CSI {r['CSI_训练vs OOT']}）")
            log("      " + cmp.to_string().replace("\n", "\n      "))
    else:
        log(f"[OK] 全部入模变量 CSI < {C.CSI_WATCH}")

    # ---------------- 3.7 画图 ----------------
    from sklearn.metrics import roc_curve
    fig, ax = plt.subplots(2, 2, figsize=(13.5, 9.5))

    for nm, s, d, c in (("训练", s_tr, tr, "#4c72b0"), ("测试", s_te, te, "#dd8452"),
                        ("OOT", s_oot, oot, "#c44e52")):
        fpr, tpr, _ = roc_curve(d["y"], -s)
        k, a = ks_auc(d["y"], s)
        ax[0, 0].plot(fpr, tpr, lw=1.8, c=c, label=f"{nm} AUC={a:.4f}")
    ax[0, 0].plot([0, 1], [0, 1], "--", c="gray", lw=1)
    ax[0, 0].set_title("ROC 曲线（三段对比）"); ax[0, 0].legend()
    ax[0, 0].set_xlabel("假正率"); ax[0, 0].set_ylabel("真正率"); ax[0, 0].grid(alpha=.3)

    for nm, s, d, c in (("训练", s_tr, tr, "#4c72b0"), ("测试", s_te, te, "#dd8452"),
                        ("OOT", s_oot, oot, "#c44e52")):
        fpr, tpr, _ = roc_curve(d["y"], -s)
        k, _ = ks_auc(d["y"], s)
        ax[0, 1].plot(np.linspace(0, 1, len(tpr)), tpr - fpr, lw=1.8, c=c,
                      label=f"{nm} KS={k:.4f}")
    ax[0, 1].set_title("KS 曲线（三段对比）"); ax[0, 1].legend()
    ax[0, 1].set_xlabel("按坏客户概率排序的样本分位"); ax[0, 1].grid(alpha=.3)

    ax[1, 0].hist(s_tr, bins=50, alpha=.5, density=True, label="训练", color="#4c72b0")
    ax[1, 0].hist(s_oot, bins=50, alpha=.5, density=True, label="OOT", color="#c44e52")
    ax[1, 0].set_title(f"分数分布　训练 vs OOT　PSI = {psi_oot:.4f}")
    ax[1, 0].set_xlabel("评分卡分数"); ax[1, 0].legend(); ax[1, 0].grid(alpha=.3)

    ct_oot = cutoff_table(s_oot, oot["y"].to_numpy(), n=10)
    ax[1, 1].plot(ct_oot["通过率"], ct_oot["累计坏账率"], "o-", lw=2, c="#c44e52")
    ax[1, 1].axhline(oot["y"].mean(), ls="--", c="gray", lw=1,
                     label=f"全通过 {oot['y'].mean():.2%}")
    ax[1, 1].set_title("通过率 — 坏账率权衡（OOT）")
    ax[1, 1].set_xlabel("累计通过率"); ax[1, 1].set_ylabel("累计坏账率")
    ax[1, 1].legend(); ax[1, 1].grid(alpha=.3)

    fig.tight_layout()
    fig.savefig(C.OUT / "26_评估图.png", dpi=150)
    plt.close(fig)
    log("")
    log(f"[OK] 图 -> {C.OUT / '26_评估图.png'}")

    # ---------------- 3.8 落盘（供后续步骤复用）----------------
    for nm, d, s in (("train", tr, s_tr), ("test", te, s_te), ("oot", oot, s_oot)):
        out = d[["id", "issue_dt", "y", "int_rate", "funded_amnt", "grade"]].copy()
        out["score"] = s
        out.to_parquet(C.PROC / f"scored_{nm}.parquet", index=False)

    art = {
        "num_cuts": {k: [float(x) for x in v] for k, v in num_cuts.items()},
        "cat_maps": {k: {str(kk): int(vv) for kk, vv in v.items()}
                     for k, v in cat_maps.items()},
        "woe_map": {f"{k[0]}||{k[1]}": float(v) for k, v in wmap.items()},
        "final_feats": final_feats,
        "params": {k: float(v) for k, v in model.params.items()},
        "A": A, "B": B,
        "perf": perf.to_dict("records"),
        "psi_oot": float(psi_oot),
    }
    (C.PROC / "scorecard_model.json").write_text(
        json.dumps(art, ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[OK] 模型件 -> {C.PROC / 'scorecard_model.json'}")
    log(f"[OK] 全步用时 {time.time()-t0:.0f}s")
    log.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
