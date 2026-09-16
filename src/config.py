# -*- coding: utf-8 -*-
"""全局路径与建模口径常量。改口径只改这一个文件。"""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
RAW = DATA / "raw"
PROC = DATA / "processed"
OUT = ROOT / "output"
DOCS = ROOT / "docs"
for _p in (PROC, OUT, DOCS):
    _p.mkdir(parents=True, exist_ok=True)

ACCEPTED_CSV = RAW / "accepted_2007_to_2018Q4.csv"
REJECTED_CSV = RAW / "rejected_2007_to_2018Q4.csv.gz"

ACCEPTED_PQ = PROC / "accepted_36m.parquet"
SAMPLE_PQ = PROC / "model_sample.parquet"
REJECTED_PQ = PROC / "rejected_window.parquet"

# ---------------------------------------------------------------- 建模口径

# 只建 36 期产品的卡。36 / 60 期的风险结构与期限结构不同，混在一起建卡是错的。
TERM = "36 months"

# 数据快照时点：accepted 表里 last_pymnt_d 最大 2019-03，last_credit_pull_d 最大 2019-04，
# 即这份数据是 2019 年 3—4 月拉的。任何 cohort 想观察到 MOB=W 的表现，
# 必须满足 放款月 + W <= 2019-03，否则它的坏账率是被右截断的假低值。
SNAPSHOT = "2019-03"

# vintage 分析用的放款区间。
# 起点定在 2016-01 是有依据的，不是随手切的：step1 查出 14 个征信字段是 LC
# 2015 年底才开始全量采集的，2015-12 之前 100% 缺失。建模期与 OOT 期必须都落在
# 采集口径一致的区间内，否则这些变量的 CSI 会因为技术原因爆表、被误读成人群漂移。
# 2016-01 放款的 cohort 到数据快照 2019-03 有 38 个月账龄，覆盖 36 期产品全周期，
# 画 vintage 曲线绰绰有余。
VINTAGE_START, VINTAGE_END = "2016-01", "2018-12"

# 表现期（MOB）。这个值由 vintage 曲线决定，step2 会把依据打印出来；
# 先给一个占位默认，跑完 step2 后按曲线回填。
PERFORM_WINDOW = 18

# 坏客户口径：观察点上连续 MONTHS_DELINQ 个月未还款 = M3+
MONTHS_DELINQ = 3

# 跨时间切分（按放款月，不是随机切）。
# 上界卡在 2017-09：再晚的放款月满足不了「放款月 + 18 个月表现期 <= 2019-03 快照」。
TRAIN_START, TRAIN_END = "2016-01", "2016-12"
OOT_START, OOT_END = "2017-01", "2017-09"
TEST_SIZE = 0.25          # 训练区间内再随机留出的测试集比例
RANDOM_STATE = 42

# ---------------------------------------------------------------- 建模参数
IV_FLOOR = 0.02           # IV 低于此值没有区分能力
IV_CEIL = 0.80            # IV 高得离谱要回头查是不是泄漏
CORR_CEIL = 0.70          # 相关系数超过此值的两个变量只留 IV 大的
VIF_CEIL = 10.0
P_CEIL = 0.05
MISSING_CEIL = 0.95       # 缺失率超过此值直接不进候选
MAX_BINS = 6
MIN_BIN_RATE = 0.05

PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50

# ---------------------------------------------------------------- 监控阈值
PSI_WATCH, PSI_ALERT = 0.10, 0.25
CSI_WATCH, CSI_ALERT = 0.10, 0.25
KS_DECAY_WATCH, KS_DECAY_ALERT = 0.10, 0.20   # 相对建模期 KS 的下滑比例
