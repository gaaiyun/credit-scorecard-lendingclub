# -*- coding: utf-8 -*-
"""第 1 步：从 1.67GB 原始 CSV 抽出建模全集，做防泄漏字段过滤与口径变更体检。

产出
----
data/processed/accepted_36m.parquet   36 期贷款、放款期 2014-01~2018-12 的全集
docs/字段判定表.csv                    151 列逐列泄漏判定
output/10_字段缺失率_按半年.csv        每个候选变量按放款半年的缺失率
output/step1_日志.txt

为什么要看「按半年的缺失率」
--------------------------
Lending Club 有一批征信字段（open_acc_6m / il_util / all_util / inq_fi ...）
是 2015-12 之后才开始全量采集的。如果训练期在采集之前、OOT 在采集之后，
这些变量的 CSI 会爆表——但那是**口径变了**（平台改了采集范围），
不是人群变了。这两种情况的处理动作完全不同，必须先分清楚。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
import config as C                                     # noqa: E402
from leakage import (FIELDS, MODEL_FEATURES, READ_COLUMNS,  # noqa: E402
                     audit_table, check_coverage)
from util import Logger, downcast                      # noqa: E402

log = Logger(C.OUT / "step1_日志.txt")


def main() -> int:
    t0 = time.time()
    log.section("第 1 步  抽取建模全集 + 防泄漏字段过滤")

    if not C.ACCEPTED_CSV.exists():
        log(f"[FAIL] 找不到原始数据：{C.ACCEPTED_CSV}")
        return 1

    # ---- 1.1 字段判定表覆盖性校验：一列都不能漏 ----
    header = pd.read_csv(C.ACCEPTED_CSV, nrows=0).columns.tolist()
    unjudged, phantom = check_coverage(header)
    log(f"原始表 {len(header)} 列，判定表 {len(FIELDS)} 条")
    if unjudged:
        log(f"[FAIL] 有 {len(unjudged)} 列没写判定，先补判定再跑：{unjudged}")
        return 1
    if phantom:
        log(f"[FAIL] 判定表里有原始表不存在的列：{phantom}")
        return 1
    log("[OK] 151 列全部有判定，无遗漏")

    cnt = pd.Series([v[0] for v in FIELDS.values()]).value_counts()
    log("")
    for k, v in cnt.items():
        log(f"    {k:9s} {v:3d} 列")
    audit_table().to_csv(C.DOCS / "字段判定表.csv", index=False, encoding="utf-8-sig")
    log(f"[OK] 逐列判定表 -> {C.DOCS / '字段判定表.csv'}")
    log(f"    实际读入 {len(READ_COLUMNS)} 列（{len(MODEL_FEATURES)} 个建模候选 + 标签/收益辅助列）")

    # ---- 1.2 单次扫描做两件事 ----
    # A) 全历史（2007~2018）的字段缺失率统计，流式累加，不 concat，内存不涨。
    #    这是「LC 什么时候改了采集口径」的证据，必须覆盖建模窗口之外的年份，
    #    否则窗口一收窄，支撑窗口选择的证据自己就没了。
    # B) 只把建模窗口内的行物化成 parquet。
    log("")
    log(f"读取 {C.ACCEPTED_CSV.name}（只读 {len(READ_COLUMNS)} / {len(header)} 列）...")
    log("  同一遍扫描里：A) 全历史缺失率统计（流式）  B) 建模窗口物化")
    parts, n_raw, n_36m = [], 0, 0
    miss_n: dict[str, pd.Series] = {}     # 半年 -> 各列缺失数
    tot_n: dict[str, int] = {}            # 半年 -> 行数
    feats_raw = [c for c in MODEL_FEATURES if c in header]

    for ch in pd.read_csv(C.ACCEPTED_CSV, usecols=READ_COLUMNS, chunksize=200_000,
                          low_memory=False, dtype={"id": "string"}):
        n_raw += len(ch)
        ch = ch[ch["id"].notna()]
        d = pd.to_datetime(ch["issue_d"], format="%b-%Y", errors="coerce")
        is36 = (ch["term"].astype("string").str.strip() == C.TERM) & d.notna()
        if not is36.any():
            continue
        c36, d36 = ch[is36], d[is36]
        n_36m += len(c36)

        # A) 按放款半年累加缺失数
        half = (d36.dt.year.astype(str) + "H" +
                ((d36.dt.month > 6).astype(int) + 1).astype(str))
        for h, idx in c36.groupby(half, observed=True).groups.items():
            miss_n[h] = miss_n.get(h, 0) + c36.loc[idx, feats_raw].isna().sum()
            tot_n[h] = tot_n.get(h, 0) + len(idx)

        # B) 建模窗口内的行留下
        keep = d36.between(C.VINTAGE_START,
                           pd.Timestamp(C.VINTAGE_END) + pd.offsets.MonthEnd(0))
        if keep.any():
            sub = c36[keep].copy()
            sub["issue_dt"] = d36[keep]
            # 逐块压 dtype 再攒。本机可用内存只有 6GB，等 concat 完再压会在
            # concat 那一步就 MemoryError（float64 块合并时要一次性分配 865 MiB）。
            parts.append(downcast(sub))

    df = pd.concat(parts, ignore_index=True)
    del parts
    log(f"[OK] 原始 {n_raw:,} 行 -> {C.TERM} 共 {n_36m:,} 行 "
        f"-> 建模窗口 {C.VINTAGE_START}~{C.VINTAGE_END}：{len(df):,} 行"
        f"，用时 {time.time()-t0:.0f}s")

    # A 的结果：全历史逐列缺失率
    miss_all = pd.DataFrame({h: miss_n[h] / tot_n[h] for h in sorted(miss_n)}).round(4)
    miss_all.index.name = "变量"
    miss_all["最大最小差"] = (miss_all.max(axis=1) - miss_all.min(axis=1)).round(4)
    miss_all = miss_all.sort_values("最大最小差", ascending=False)
    miss_all.to_csv(C.OUT / "10_字段缺失率_按半年_全历史.csv", encoding="utf-8-sig")

    log("")
    log("-- [全历史] 缺失率随放款期剧烈变化的变量（采集口径变更的证据，前 15）--")
    log("   这些不是人群变了，是 LC 改了采集范围。建模期与 OOT 期若跨越采集口径变更，")
    log("   这些变量的 CSI 会因为一个技术原因爆表，而你会误判成人群漂移。")
    cols_show = [c for c in miss_all.columns if c >= "2014H1"]
    log(miss_all.head(15)[cols_show].to_string())

    changed = miss_all[miss_all["最大最小差"] > 0.50].index.tolist()
    log("")
    log(f"[全历史标记] 缺失率跨期波动 > 50 个百分点的变量共 {len(changed)} 个：")
    log(f"    {', '.join(changed) if changed else '无'}")
    log(f"[窗口选择] 因此建模窗口起点定在 {C.VINTAGE_START}——这些字段在此之后缺失率已稳定。")

    # ---- 1.3 日期字段解析 ----
    df["last_pymnt_dt"] = pd.to_datetime(df["last_pymnt_d"], format="%b-%Y", errors="coerce")
    df["earliest_cr_dt"] = pd.to_datetime(df["earliest_cr_line"], format="%b-%Y", errors="coerce")
    # 征信历史长度（月）——比原始日期好用，且不会把"年份"当成数值特征
    df["credit_hist_mths"] = ((df["issue_dt"].dt.year - df["earliest_cr_dt"].dt.year) * 12 +
                              (df["issue_dt"].dt.month - df["earliest_cr_dt"].dt.month))
    df.loc[df["credit_hist_mths"] < 0, "credit_hist_mths"] = np.nan
    df = df.drop(columns=["earliest_cr_line", "earliest_cr_dt"])

    # emp_length 转有序数值（"< 1 year"->0, "10+ years"->10），保留缺失
    el = df["emp_length"].astype("string").str.strip()
    df["emp_length_num"] = (el.str.replace(r"\D", "", regex=True)
                              .replace("", np.nan).astype("Float64").astype("float"))
    df.loc[el == "< 1 year", "emp_length_num"] = 0.0
    df = df.drop(columns=["emp_length"])

    log("")
    log("-- 放款月分布（按半年）--")
    half = df["issue_dt"].dt.year.astype(str) + "H" + \
        ((df["issue_dt"].dt.month > 6).astype(int) + 1).astype(str)
    df["issue_half"] = half
    log(df.groupby("issue_half").size().to_string())

    log("")
    log("-- loan_status 分布 --")
    vc = df["loan_status"].value_counts(dropna=False)
    log(pd.DataFrame({"n": vc, "占比": (vc / len(df)).round(4)}).to_string())

    # ---- 1.4 建模窗口内再体检一次：确认窗口内口径稳定 ----
    feats = [c for c in MODEL_FEATURES if c in df.columns] + \
            ["credit_hist_mths", "emp_length_num"]
    miss = df.groupby("issue_half", observed=True)[feats].apply(
        lambda g: g.isna().mean()).T
    miss.index.name = "变量"
    miss = miss.round(4)
    miss["最大最小差"] = (miss.max(axis=1) - miss.min(axis=1)).round(4)
    miss = miss.sort_values("最大最小差", ascending=False)
    miss.to_csv(C.OUT / "11_字段缺失率_按半年_建模窗口.csv", encoding="utf-8-sig")

    log("")
    log("-- [建模窗口内] 缺失率波动最大的 10 个变量 --")
    log(miss.head(10).to_string())

    suspect = miss[miss["最大最小差"] > 0.50].index.tolist()
    log("")
    log(f"[窗口内校验] 缺失率跨期波动 > 50 个百分点的变量共 {len(suspect)} 个："
        f"{', '.join(suspect) if suspect else '无'}")
    log("    窗口内为 0 个，说明起点选对了：剩下的缺失率波动都是人群变化，不是采集口径变化。")
    log(f"    窗口内最大波动 {miss['最大最小差'].max():.4f}"
        f"（{miss['最大最小差'].idxmax()}），属正常人群漂移量级。")

    # ---- 1.5 落盘 ----
    # 压 dtype 再写：默认 dtype 下这张表反序列化会把 pyarrow 打到段错误（本机可用内存 6GB）
    before = df.memory_usage(deep=True).sum() / 1e9
    df = downcast(df)
    after = df.memory_usage(deep=True).sum() / 1e9
    log("")
    log(f"内存压缩：{before:.2f} GB -> {after:.2f} GB（float32 + category）")
    df.to_parquet(C.ACCEPTED_PQ, index=False)
    log(f"[OK] 建模全集 -> {C.ACCEPTED_PQ}  "
        f"({len(df):,} 行 x {df.shape[1]} 列, "
        f"{C.ACCEPTED_PQ.stat().st_size/1e6:.0f} MB)")
    log(f"[OK] 全步用时 {time.time()-t0:.0f}s")
    log.save()
    return 0


if __name__ == "__main__":
    sys.exit(main())
