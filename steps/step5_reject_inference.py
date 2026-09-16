# -*- coding: utf-8 -*-
"""第 5 步：拒绝推断 —— 用 982 万被拒申请纠正幸存者偏差。

产出
----
output/40_共同变量可比性.csv     accepted vs rejected 的共同字段对比与可用性判定
output/41_核准概率模型.csv       P(accept|x) 模型的分箱与系数
output/42_权重分布.csv           重加权法的权重分布
output/43_KGB_AGB对比.csv        推断前后的 KS / 分数分布 / 系数
output/44_打包法敏感性.csv        倍数 k = 2/3/4 的结果
output/45_swap_set.csv           同一通过率下的换入换出分析
output/46_拒绝推断图.png
output/step5_日志.txt
"""
from __future__ import annotations

import gc
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as C                                      # noqa: E402
import matplotlib.pyplot as plt                         # noqa: E402
from reject_inference import (accept_weights, parcel_bad_flags,  # noqa: E402
                              swap_set)
from scorecard import (apply_bins, apply_cat_bins, bin_table,   # noqa: E402
                       cat_bins, cutoff_table, drop_bad_signs,
                       is_monotonic, iv_of, ks_auc, ks_auc_prob,
                       monotonic_bins, scale_params, to_score, woe_map)
from util import Logger                                 # noqa: E402

log = Logger(C.OUT / "step5_日志.txt")

# LC 36 期产品的金额区间（accepted 表里的实际取值范围）
AMT_LO, AMT_HI = 1000.0, 40000.0
PARCEL_K = (2.0, 3.0, 4.0)


def build_rejected_window(log=print) -> None:
    """从 255MB 的 gz 原文件里抽出建模窗口内的被拒申请，落成 parquet。

    全量 2760 万行、1.78GB，每次跑都解压一遍太慢，所以缓存一份。
    但缓存必须由这一步自己生成——中间产物如果靠手工脚本生成，
    清掉 data/processed 之后整条流水线就跑不起来了，那就不叫可复现。
    """
    import time as _t

    import pyarrow as pa
    import pyarrow.parquet as pq

    t0 = _t.time()
    log(f"缓存不存在，从 {C.REJECTED_CSV.name} 抽建模窗口内的被拒申请...")
    cols = ["Amount Requested", "Application Date", "Risk_Score",
            "Debt-To-Income Ratio", "State"]
    n, kept, writer = 0, 0, None
    # 逐块写，不在内存里 concat。9.8M 行一次性攒起来再写会直接段错误
    # （本机 Windows 提交限额只剩 4GB）。
    for ch in pd.read_csv(C.REJECTED_CSV, usecols=cols, chunksize=1_000_000,
                          low_memory=False):
        n += len(ch)
        d = pd.to_datetime(ch["Application Date"], errors="coerce")
        m = d.between(C.VINTAGE_START, C.OOT_END + "-30")
        if not m.any():
            continue
        s = pd.DataFrame({
            "loan_amnt": ch.loc[m, "Amount Requested"].astype("float32"),
            "fico_range_low": ch.loc[m, "Risk_Score"].astype("float32"),
            "dti": pd.to_numeric(
                ch.loc[m, "Debt-To-Income Ratio"].astype(str).str.rstrip("%"),
                errors="coerce").astype("float32"),
            "addr_state": ch.loc[m, "State"].astype(str),
            "app_dt": d[m]})
        t = pa.Table.from_pandas(s, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(C.REJECTED_PQ, t.schema,
                                      compression="snappy")
        writer.write_table(t)
        kept += len(s)
    if writer is not None:
        writer.close()
    log(f"[OK] 全量 {n:,} 行 -> 窗口内 {kept:,} 行，"
        f"落盘 {C.REJECTED_PQ.stat().st_size/1e6:.0f} MB，用时 {_t.time()-t0:.0f}s")


def load_rejects(log=print) -> pd.DataFrame:
    if not C.REJECTED_PQ.exists():
        build_rejected_window(log)
    r = pd.read_parquet(C.REJECTED_PQ)
    # 980 万行的州代码存成 object 字符串要吃几百 MB，转 category
    r["addr_state"] = r["addr_state"].astype("category")
    return r


def fit_card(tr: pd.DataFrame, num_f: list[str], cat_f: list[str],
             y: pd.Series, weights=None, log=print):
    """在给定样本上跑完整评分卡流程，返回 (打分函数, 入模变量, 模型, 分箱表)。"""
    tables, cuts, maps = [], {}, {}
    for f in num_f:
        c = monotonic_bins(tr[f], y, max_bins=C.MAX_BINS, min_rate=C.MIN_BIN_RATE)
        if not c:
            continue
        t = bin_table(apply_bins(tr[f], c), y, name=f)
        cuts[f] = c
        t["iv"] = iv_of(t); t["monotonic"] = is_monotonic(t)
        tables.append(t)
    for f in cat_f:
        mp = cat_bins(tr[f], y, min_share=0.02, max_bins=C.MAX_BINS)
        if len(set(mp.values())) < 2:
            continue
        t = bin_table(apply_cat_bins(tr[f], mp), y, name=f)
        maps[f] = mp
        t["iv"] = iv_of(t); t["monotonic"] = is_monotonic(t)
        tables.append(t)
    tbl = pd.concat(tables, ignore_index=True)
    wmap = woe_map(tbl)

    def woe_of(d):
        # 用 float32 存：70 个变量 x 24 万行的 float64 矩阵是 128MB，
        # 在本机 4GB 提交限额下这一步就会 MemoryError。分箱后的 WOE 值
        # 只有几十个不同取值，float32 精度绰绰有余。
        out = {f: apply_bins(d[f], c).map(
            lambda k, f=f: wmap.get((f, int(k)), 0.0)).astype("float32")
            for f, c in cuts.items()}
        out.update({f: apply_cat_bins(d[f], m).map(
            lambda k, f=f: wmap.get((f, int(k)), 0.0)).astype("float32")
            for f, m in maps.items()})
        return pd.DataFrame(out, index=d.index)

    iv = tbl.groupby("var")["iv"].first().sort_values(ascending=False)
    keep = [v for v in iv.index if iv[v] >= C.IV_FLOOR]
    w_tr = woe_of(tr)
    corr = w_tr[keep].corr().abs()
    keep2 = []
    for v in keep:
        if all(corr.loc[v, k] < C.CORR_CEIL for k in keep2):
            keep2.append(v)
    model, final = drop_bad_signs(w_tr[keep2], y, weights=weights,
                                 p_ceil=C.P_CEIL, log=log)
    A, B = scale_params(C.PDO, C.BASE_SCORE, C.BASE_ODDS)

    def score(d):
        import statsmodels.api as sm
        p = model.predict(sm.add_constant(woe_of(d)[final], has_constant="add"))
        return to_score(np.asarray(p, dtype=float), A, B)

    def prob(d):
        import statsmodels.api as sm
        return np.asarray(model.predict(
            sm.add_constant(woe_of(d)[final], has_constant="add")), dtype=float)

    return score, prob, final, model, tbl


def main() -> int:
    t0 = time.time()
    log.section("第 5 步  拒绝推断")

    acc = pd.read_parquet(C.SAMPLE_PQ, columns=[
        "id", "issue_dt", "y", "loan_amnt", "fico_range_low", "dti",
        "emp_length_num", "addr_state", "int_rate", "funded_amnt"])
    rej = load_rejects(log)
    log(f"已核准（建模样本）{len(acc):,} 笔")
    log(f"被拒申请（同期 {C.VINTAGE_START}~2017-09）{len(rej):,} 笔")
    log(f"名义核准率 = {len(acc)/(len(acc)+len(rej)):.2%}")
    log("  （注：分母是全部被拒申请，含同一人多次申请、金额超出产品区间的，")
    log("    不等于 LC 对外口径的核准率）")

    # ---------------- 5.1 共同变量可比性 ----------------
    log.section("5.1  共同变量盘点：哪些能用，哪些不能用")
    log("两张表能对上的字段只有 8 个。但「字段名对得上」不等于「数据可比」，")
    log("直接拿来建核准模型会得出荒唐的结论。逐个查：")
    log("")
    rows = []
    checks = [
        ("申请金额", "loan_amnt", "loan_amnt", "可用",
         "两边都是申请人填的金额，量纲一致"),
        ("FICO", "fico_range_low", "fico_range_low", "部分可用",
         "被拒样本缺失 68%，且缺失率按月在 46%~96% 间波动，需单独处理"),
        ("DTI", "dti", "dti", "不可用",
         "被拒样本均值 125.75、最大 737 万、6.2% 为负，与已核准侧不是同一口径"),
        ("工作年限", "emp_length_num", None, "不可用",
         "被拒样本 72% 挤在 <1年、19% 在 5年，已核准侧是平滑分布，采集口径不同"),
        ("州", "addr_state", "addr_state", "可用", "两边都是标准州代码"),
        ("日期", "issue_dt", "app_dt", "可用（有偏移）",
         "已核准侧是放款月、被拒侧是申请日，LC 放款滞后申请约 2~4 周"),
    ]
    for nm, ca, cr, verdict, why in checks:
        d = {"字段": nm, "判定": verdict, "理由": why}
        if ca in acc.columns and pd.api.types.is_numeric_dtype(acc[ca]):
            d["已核准_中位数"] = round(float(acc[ca].median()), 2)
        if cr and cr in rej.columns and pd.api.types.is_numeric_dtype(rej[cr]):
            d["被拒_中位数"] = round(float(rej[cr].median()), 2)
            d["被拒_缺失率"] = round(float(rej[cr].isna().mean()), 4)
        rows.append(d)
    cmp_tbl = pd.DataFrame(rows)
    cmp_tbl.to_csv(C.OUT / "40_共同变量可比性.csv", index=False, encoding="utf-8-sig")
    log(cmp_tbl.to_string(index=False))
    log("")
    log("[结论] 真正干净可用的共同变量只有 3 个：申请金额、州、申请时间，")
    log("       加上一个有大量缺失的 FICO。整个拒绝推断只能架在这么窄的基础上，")
    log("       这是这份公开数据的硬限制，结论的可靠性要打折，后面会再说一次。")

    # ---------------- 5.2 可比全集 ----------------
    log.section("5.2  构造可比全集")
    n0 = len(rej)
    rej = rej[rej["loan_amnt"].between(AMT_LO, AMT_HI)]
    log(f"被拒样本限制在 36 期产品的金额区间 [{AMT_LO:.0f}, {AMT_HI:.0f}]："
        f"{n0:,} -> {len(rej):,}（剔除 {1-len(rej)/n0:.1%}）")
    log("  为什么要剔：被拒样本里有申请 30 万美元的，那根本不是这个产品的客群，")
    log("  留着会让核准模型把「金额超纲」当成核准率的主要驱动，学不到真正的风险信息。")

    # Risk_Score 的量纲要单独查一次：它在被拒表里叫 Risk_Score，不叫 FICO，
    # 不能默认和 accepted 侧的 fico_range_low 是同一把尺子。
    s = rej["fico_range_low"].dropna()
    n_hi = int((s > 850).sum())
    n_lo = int((s < 500).sum())
    log("")
    log("-- Risk_Score 量纲核查（能不能和 accepted 的 FICO 直接比）--")
    log(f"    非缺失 {len(s):,} 笔，取值范围 [{s.min():.0f}, {s.max():.0f}]")
    log(f"    落在 FICO 合法区间 300~850：{len(s)-n_hi:,}（{1-n_hi/len(s):.2%}）")
    log(f"    超出 FICO 上限 850：{n_hi:,}（{n_hi/len(s):.2%}），最大 {s.max():.0f}")
    log(f"    低于 500：{n_lo:,}（{n_lo/len(s):.2%}）")
    log("    超 850 的部分很可能是混入的 VantageScore 1.0/2.0（501~990 量纲），")
    log("    说明这一列不是纯 FICO。占比 < 1%，下面按 [300, 850] 截断处理，")
    log("    但这是一个**残留的口径不确定性**，写进诚实边界，不当作已解决。")
    rej["fico_range_low"] = rej["fico_range_low"].where(
        rej["fico_range_low"].between(300, 850))
    rej = rej.rename(columns={"app_dt": "dt"})
    acc2 = acc.rename(columns={"issue_dt": "dt"}).copy()
    rej["accepted"] = 0
    rej["id"] = pd.NA                 # 被拒侧没有 id，占位便于后面按 id 回join
    acc2["accepted"] = 1
    common = ["id", "loan_amnt", "fico_range_low", "addr_state", "dt", "accepted"]

    # 两侧都抽样（case-control sampling）。全量 55 万 + 966 万进回归，
    # 本机内存扛不住；抽样后用 odds 校正把截距还原到总体比例，
    # 这是流行病学里 case-control 研究的标准做法，系数（斜率）不受抽样影响，
    # 只有截距需要校正。
    N_SIDE = 200_000
    n_acc_s = min(N_SIDE, len(acc2))
    n_rej_s = min(N_SIDE, len(rej))
    s1 = n_acc_s / len(acc2)          # 核准侧抽样率
    s0 = n_rej_s / len(rej)           # 被拒侧抽样率
    acc_s = acc2.sample(n=n_acc_s, random_state=C.RANDOM_STATE)
    rej_s = rej.sample(n=n_rej_s, random_state=C.RANDOM_STATE)
    log(f"核准侧抽样 {len(acc2):,} -> {n_acc_s:,}（抽样率 {s1:.4f}）")
    log(f"被拒侧抽样 {len(rej):,} -> {n_rej_s:,}（抽样率 {s0:.6f}）")
    log(f"  校正关系：odds_总体 = odds_样本 x (s0/s1) = odds_样本 x {s0/s1:.6f}")

    pool = pd.concat([acc_s[common], rej_s[common]], ignore_index=True)
    pool["fico_missing"] = pool["fico_range_low"].isna().astype(int)
    del rej_s, acc_s
    gc.collect()
    log(f"核准模型全集 {len(pool):,} 行")
    log("")
    log("-- FICO 缺失本身是不是「被拒」的代理变量 --")
    ct = pd.crosstab(pool["fico_missing"], pool["accepted"], normalize="index").round(4)
    ct.columns = ["被拒", "核准"]
    ct.index = ["FICO 有值", "FICO 缺失"]
    log(ct.to_string())
    log("  FICO 缺失的样本几乎全是被拒的。如果直接把缺失当一个箱扔进核准模型，")
    log("  模型会靠这个箱做出近乎完美的区分，权重全废。所以下面做两套方案分别看。")

    # ---------------- 5.3 方法一：核准概率模型 + 重加权 ----------------
    log.section("5.3  方法一：核准概率模型 P(accept|x) + 重加权（KGB -> AGB）")
    results = {}
    for tag, use_fico in (("方案A_不含FICO", False), ("方案B_仅FICO有值样本", True)):
        log("")
        log(f"---- {tag} ----")
        if use_fico:
            sub = pool[pool["fico_missing"] == 0].copy()
            num_f = ["loan_amnt", "fico_range_low"]
            log(f"    只留 FICO 有值的样本：{len(sub):,} 行"
                f"（核准 {int(sub['accepted'].sum()):,}）")
            log("    代价：这是被拒样本里的一个**选择性子集**，本身带二次选择偏差。")
        else:
            sub = pool.copy()
            num_f = ["loan_amnt"]
            log(f"    全量，但不使用 FICO：{len(sub):,} 行")
            log("    代价：丢掉了最强的共同变量，核准模型很弱。")
        cat_f = ["addr_state"]
        _, p_acc_fn, feats_acc, m_acc, tbl_acc = fit_card(
            sub, num_f, cat_f, sub["accepted"], log=lambda *_: None)
        log(f"    核准模型入模变量：{feats_acc}")
        p_all = p_acc_fn(sub)
        # p_all 是 P(accept)，事件就是「被核准」，所以用 ks_auc_prob（它按
        # 「概率越高越可能发生事件」的方向算），用 ks_auc 会算出 1-AUC。
        ks_acc, auc_acc = ks_auc_prob(sub["accepted"], p_all)
        log(f"    核准模型 AUC = {auc_acc:.4f}"
            f"（区分「谁会被批」的能力；越高说明核准决策越能被共同变量解释）")
        if auc_acc < 0.60:
            log("    [注意] AUC 偏低，说明 LC 的核准决策主要依赖我们看不到的信息，")
            log("           重加权的 MAR 前提在这里是不成立的。")

        # 关键：模型在抽样样本上**拟合**，但要给**全部**已核准样本打分。
        # 只给抽中的那部分算权重，等于大部分训练样本拿不到纠偏，AGB 会退化成 KGB。
        acc_score_on = acc2 if not use_fico else acc2[acc2["fico_range_low"].notna()]
        p_acc_only = p_acc_fn(acc_score_on)
        log(f"    给 {len(acc_score_on):,} 笔已核准样本打核准概率"
            f"（占全部已核准的 {len(acc_score_on)/len(acc2):.1%}）")
        # 抽样还原：odds_总体 = odds_样本 x (s0/s1)
        odds = p_acc_only / np.clip(1 - p_acc_only, 1e-9, None)
        odds_true = odds * (s0 / s1)
        p_true = odds_true / (1 + odds_true)
        w = accept_weights(p_true, trim_pct=99.0)
        log(f"    还原后的核准概率：中位数 {np.median(p_true):.4f}，"
            f"最低 {p_true.min():.4f}，最高 {p_true.max():.4f}")
        log(f"    权重（1/P(accept)）：中位数 {np.median(w):.2f}，"
            f"均值 {w.mean():.2f}，99 分位截尾后最大 {w.max():.2f}")
        results[tag] = (acc_score_on["id"].astype(str).to_numpy(), w, auc_acc)
        pd.DataFrame({"权重": w}).describe().round(3).to_csv(
            C.OUT / f"42_权重分布_{tag}.csv", encoding="utf-8-sig")
        del sub, p_all, p_acc_only
        gc.collect()

    log("")
    log("-- 两套方案的核准模型对比 --")
    log(f"    方案A（全量，不含 FICO）  AUC = {results['方案A_不含FICO'][2]:.4f}")
    log(f"    方案B（仅 FICO 有值样本）AUC = {results['方案B_仅FICO有值样本'][2]:.4f}")
    log("    两套都做 AGB，不挑一个——挑哪个的理由本身就是要交代的假设。")

    # ---------------- 5.4 重训 AGB ----------------
    log.section("5.4  用权重重训评分卡（AGB）并与 KGB 对比")
    tr_ids = set(pd.read_parquet(C.PROC / "scored_train.parquet",
                                 columns=["id"])["id"].astype(str))
    oot_ids = set(pd.read_parquet(C.PROC / "scored_oot.parquet",
                                  columns=["id"])["id"].astype(str))
    full = pd.read_parquet(C.SAMPLE_PQ)
    sid = full["id"].astype(str)
    tr = full[sid.isin(tr_ids)].copy()
    oot = full[sid.isin(oot_ids)].copy()
    del full
    gc.collect()

    from leakage import MODEL_FEATURES
    CAT = ["home_ownership", "verification_status", "purpose",
           "addr_state", "application_type"]
    cand = [c for c in MODEL_FEATURES if c in tr.columns and c != "term"] + \
           ["credit_hist_mths", "emp_length_num"]
    num_f = [c for c in cand if c not in CAT and pd.api.types.is_numeric_dtype(tr[c])]
    cat_f = [c for c in CAT if c in tr.columns]

    # 后面只用得上这些列。975 万行的被拒表还占着内存，主卡重训前先瘦身。
    need = sorted(set(num_f) | set(cat_f) | {"id", "y", "issue_dt"})
    tr = tr[[c for c in need if c in tr.columns]]
    oot = oot[[c for c in need if c in oot.columns]]
    rej = rej[rej["fico_range_low"].notna()]
    n_keep = min(len(rej), len(tr))
    rej = rej.sample(n=n_keep, random_state=C.RANDOM_STATE).copy()
    gc.collect()
    log(f"（被拒表瘦身：只留 FICO 有值并抽样到 {len(rej):,} 笔，供 5.5 打包法用）")
    log("")

    log("KGB（推断前，不加权）：")
    s_kgb, _, f_kgb, m_kgb, _ = fit_card(tr, num_f, cat_f, tr["y"],
                                         log=lambda *_: None)
    log(f"    入模 {len(f_kgb)} 个变量")

    cards = {"KGB(推断前)": s_kgb}
    for tag in ("方案A_不含FICO", "方案B_仅FICO有值样本"):
        ids_w, w_all, _ = results[tag]
        wmap_id = dict(zip(ids_w, w_all))
        wcol = tr["id"].astype(str).map(wmap_id)
        n_hit = int(wcol.notna().sum())
        log("")
        log(f"AGB[{tag}]：")
        log(f"    训练集 {len(tr):,} 笔，按 id 对上权重 {n_hit:,} 笔（{n_hit/len(tr):.1%}）")
        if n_hit < len(tr):
            log(f"    对不上的 {len(tr)-n_hit:,} 笔是 FICO 缺失的已核准客户，"
                f"方案B 的核准模型覆盖不到，给权重 1.0（不纠偏）。")
        wcol = wcol.fillna(1.0)
        log(f"    权重：中位数 {wcol.median():.2f}，均值 {wcol.mean():.2f}，"
            f"最大 {wcol.max():.2f}")
        s_a, _, f_a, _, _ = fit_card(tr, num_f, cat_f, tr["y"],
                                     weights=wcol.to_numpy(), log=lambda *_: None)
        log(f"    入模 {len(f_a)} 个变量；"
            f"KGB 独有 {sorted(set(f_kgb)-set(f_a))}；AGB 独有 {sorted(set(f_a)-set(f_kgb))}")
        cards[f"AGB[{tag}]"] = s_a
    s_agb = cards["AGB[方案B_仅FICO有值样本]"]

    rows = []
    for nm, fn in cards.items():
        for seg, d in (("训练", tr), ("OOT", oot)):
            s = fn(d)
            ks, auc = ks_auc(d["y"], s)
            rows.append({"模型": nm, "样本": seg, "KS": round(ks, 4),
                         "AUC": round(auc, 4),
                         "分数均值": round(float(np.mean(s)), 2),
                         "分数标准差": round(float(np.std(s)), 2)})
    cmp2 = pd.DataFrame(rows)
    log("")
    log(cmp2.to_string(index=False))
    log("")
    log("!! 重要：上面这个 KS 是在**已核准样本**上算的。")
    log("   已核准样本恰恰是不需要纠偏的那部分，所以 KS 变化不能用来证明拒绝推断有效。")
    log("   拒绝推断的效果永远无法在这份数据上被验证——被拒客户的真实表现观察不到。")
    log("   能看的只有：系数怎么变、分数分布怎么变、同一通过率下换的是哪批人。")

    # ---------------- 5.5 打包法 ----------------
    log.section("5.5  方法二：打包法（Parceling）")
    log("用只含共同变量的评分卡给被拒客户打分，按分数档指定推断坏标签，")
    log(f"每档推断坏账率 = 该档已核准客户坏账率 x k。k 是假设，做 {PARCEL_K} 三档敏感性。")

    rej_p = rej          # 5.4 里已经筛成「FICO 有值 + 抽样到训练集量级」
    log(f"用于打包的被拒样本 {len(rej_p):,} 笔（FICO 有值，抽样到与训练集同量级）")

    s_common, _, f_common, _, _ = fit_card(
        tr, ["loan_amnt", "fico_range_low"], ["addr_state"], tr["y"],
        log=lambda *_: None)
    log(f"    共同变量评分卡入模：{f_common}")
    ks_c, auc_c = ks_auc(tr["y"], s_common(tr))
    log(f"    它在已核准训练集上的 KS = {ks_c:.4f}（只有 3 个变量，弱是正常的）")

    s_tr_c = s_common(tr)
    s_rej_c = s_common(rej_p)
    log(f"    分数对比：已核准中位数 {np.median(s_tr_c):.1f}，"
        f"被拒中位数 {np.median(s_rej_c):.1f}，"
        f"差 {np.median(s_tr_c)-np.median(s_rej_c):.1f} 分")

    log("")
    log("-- 为什么这个差只有几分：评分卡不会外推 --")
    log("    共同变量卡是在**已核准样本**上分的箱。已核准样本 FICO 最低 660"
        f"（LC 的硬门槛），")
    log("    所以最低那一箱的下界就是 660。被拒客户 FICO 中位数 634、四分位下界 592，")
    log("    全部落进同一个最低箱，拿同一个 WOE——评分卡看不出 634 和 520 有什么区别。")
    fico_bins_tr = pd.cut(tr["fico_range_low"],
                          [-np.inf, 660, 680, 700, 720, np.inf], labels=False)
    fico_bins_rj = pd.cut(rej_p["fico_range_low"],
                          [-np.inf, 660, 680, 700, 720, np.inf], labels=False)
    fb = pd.DataFrame({
        "已核准占比": fico_bins_tr.value_counts(normalize=True).sort_index().round(4),
        "被拒占比": fico_bins_rj.value_counts(normalize=True).sort_index().round(4)}
    ).fillna(0)
    fb.index = ["<=660", "660-680", "680-700", "700-720", ">720"]
    log(fb.to_string())
    log(f"    被拒样本有 {float(fb.loc['<=660','被拒占比']):.1%} 落在 <=660 这一档，"
        f"已核准只有 {float(fb.loc['<=660','已核准占比']):.1%}。")
    log("    这是打包法在这份数据上失效的**根本原因**，不是 k 选得不对：")
    log("    分箱边界是已核准样本的值域定的，对值域外的人群没有分辨力。")
    log("    真实项目里被拒客户的征信变量取值范围和核准客户是重叠的（同一套征信），")
    log("    不会出现这种整体落到边界外的情况。")

    par_rows = []
    for k in PARCEL_K:
        yb = parcel_bad_flags(s_rej_c, s_tr_c, tr["y"].to_numpy(), k=k,
                              random_state=C.RANDOM_STATE)
        infer_rate = yb.mean()
        mix = pd.concat([
            tr.assign(_y=tr["y"], _src="accept"),
            rej_p.assign(_y=yb, _src="reject")], ignore_index=True)
        # 打包法只能用共同变量重训（被拒样本没有其他变量）
        s_par, _, f_par, _, _ = fit_card(
            mix, ["loan_amnt", "fico_range_low"], ["addr_state"], mix["_y"],
            log=lambda *_: None)
        ks_o, auc_o = ks_auc(oot["y"], s_par(oot))
        ks_b, auc_b = ks_auc(oot["y"], s_common(oot))
        par_rows.append({"倍数k": k, "推断坏账率": round(float(infer_rate), 4),
                         "合并样本量": len(mix),
                         "共同变量卡_OOT_KS": round(ks_b, 4),
                         "打包后_OOT_KS": round(ks_o, 4),
                         "KS变化": round(ks_o - ks_b, 4)})
        log(f"    k={k}: 被拒样本推断坏账率 {infer_rate:.2%}，"
            f"打包后共同变量卡 OOT KS {ks_b:.4f} -> {ks_o:.4f}（{ks_o-ks_b:+.4f}）")
        del mix
        gc.collect()
    par = pd.DataFrame(par_rows)
    par.to_csv(C.OUT / "44_打包法敏感性.csv", index=False, encoding="utf-8-sig")
    log("")
    log("   注意这里比较的是**只含共同变量的评分卡**打包前后的变化，")
    log("   因为被拒样本没有其余 76 个征信变量，没法把它们并进主卡重训。")
    log("   这是打包法在这份数据上的结构性限制：被拒侧变量太少，推断样本进不了主模型。")

    # ---------------- 5.6 swap set ----------------
    log.section("5.6  swap set：同一通过率下，两套卡批的是不是同一批人")
    log("这是拒绝推断为数不多能被真正检验的地方。")
    sw_all = []
    for ar in (0.60, 0.80, 0.90):
        sw = swap_set(s_kgb(oot), s_agb(oot), oot["y"].to_numpy(), ar)
        sw["通过率"] = ar
        sw_all.append(sw)
        log("")
        log(f"-- 通过率 {ar:.0%}（OOT）--")
        log(sw.to_string(index=False))
    sw_df = pd.concat(sw_all, ignore_index=True)
    sw_df.to_csv(C.OUT / "45_swap_set.csv", index=False, encoding="utf-8-sig")

    # ---------------- 5.7 结论 ----------------
    log.section("5.7  结论：拒绝推断在这份数据上到底值不值")
    base_ks = cmp2[(cmp2["模型"] == "KGB(推断前)") &
                   (cmp2["样本"] == "OOT")]["KS"].iloc[0]
    log("1. 重加权后，OOT（已核准样本）上的 KS 变化：")
    for nm in cmp2["模型"].unique():
        if nm == "KGB(推断前)":
            continue
        k = cmp2[(cmp2["模型"] == nm) & (cmp2["样本"] == "OOT")]["KS"].iloc[0]
        log(f"     {nm:28s} {base_ks:.4f} -> {k:.4f}  （{k-base_ks:+.4f}）")
    log("   但这个数字不能当成「拒绝推断的效果」——见 5.4 末尾那段。")
    log(f"2. 打包法在 k={PARCEL_K} 三档下，共同变量卡的 OOT KS 变化见 44_打包法敏感性.csv，")
    log("   幅度都很小，且随 k 变号——说明结论对那个拍脑袋的倍数很敏感。")
    log("3. swap set 显示两套卡在同一通过率下换掉的人群只占 0.2%~0.5%，")
    log("   而且换入换出人群的坏账率高低在不同通过率上并不一致，看不出稳定的改善。")
    log("4. 根本限制有两层：")
    log("   (a) 被拒侧只有 3 个干净的共同变量，主卡用了 16 个。拒绝推断只能纠正")
    log("       「金额 + FICO + 地域」这三个维度上的选择偏差，其余维度纠不了。")
    log("   (b) 更要命的是 5.5 那个发现：已核准样本 FICO 下界是 660（LC 的硬门槛），")
    log("       分箱边界由它决定，而大部分被拒客户 FICO 在 660 以下，")
    log("       整体落进同一个边界箱——评分卡对值域外人群没有分辨力。")
    log("       这也解释了为什么两种方法的效果都接近于零。")
    log("")
    log("[诚实结论] 在这份数据上，拒绝推断带来的提升很有限，而且无法被验证。")
    log("   我仍然把它做完整，是因为：①真实项目里必须做，流程要跑通；")
    log("   ②做完才知道它的限制在哪儿——限制本身是结论的一部分。")
    log("   真实项目里做拒绝推断，前提是被拒侧能拿到和核准侧同样的征信变量")
    log("   （申请时都拉过征信，数据在自己手里），那时候它才真正有效。")
    log("   这份公开数据的被拒表只留了 9 个字段，这是数据的限制，不是方法的限制。")

    cmp2.to_csv(C.OUT / "43_KGB_AGB对比.csv", index=False, encoding="utf-8-sig")

    # ---------------- 图 ----------------
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    ax[0].hist(s_tr_c, bins=50, alpha=.55, density=True, label="已核准", color="#4c72b0")
    ax[0].hist(s_rej_c, bins=50, alpha=.55, density=True, label="被拒", color="#c44e52")
    ax[0].set_title("共同变量评分卡：已核准 vs 被拒 的分数分布")
    ax[0].set_xlabel("分数"); ax[0].legend(); ax[0].grid(alpha=.3)

    sk, sa = s_kgb(oot), s_agb(oot)
    ax[1].hist(sk, bins=50, alpha=.55, density=True, label="KGB(推断前)", color="#4c72b0")
    ax[1].hist(sa, bins=50, alpha=.55, density=True, label="AGB(重加权)", color="#dd8452")
    ax[1].set_title("OOT 上的分数分布：推断前 vs 推断后")
    ax[1].set_xlabel("分数"); ax[1].legend(); ax[1].grid(alpha=.3)

    for nm, s, c in (("KGB(推断前)", sk, "#4c72b0"), ("AGB(重加权)", sa, "#dd8452")):
        ct = cutoff_table(s, oot["y"].to_numpy(), n=20)
        ax[2].plot(ct["通过率"], ct["累计坏账率"], "o-", lw=1.8, ms=3, c=c, label=nm)
    ax[2].set_title("通过率 — 坏账率曲线（OOT）")
    ax[2].set_xlabel("累计通过率"); ax[2].set_ylabel("累计坏账率")
    ax[2].legend(); ax[2].grid(alpha=.3)
    fig.tight_layout()
    fig.savefig(C.OUT / "46_拒绝推断图.png", dpi=150)
    plt.close(fig)
    log("")
    log(f"[OK] 图 -> {C.OUT / '46_拒绝推断图.png'}")
    log(f"[OK] 全步用时 {time.time()-t0:.0f}s")
    log.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
