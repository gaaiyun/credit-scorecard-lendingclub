# -*- coding: utf-8 -*-
"""一条命令跑完全流程：从 1.67GB 原始 CSV 到全部产物。

用法
----
    python run_all.py                # 跑全部 7 步
    python run_all.py --from 3       # 从第 3 步开始（前面的中间结果已在 data/processed）
    python run_all.py --only 5       # 只跑第 5 步
    python run_all.py --check        # 只检查数据是否就位，不跑

数据准备（两个文件都不进 git，见 .gitignore）
    data/raw/accepted_2007_to_2018Q4.csv        1.67 GB
    data/raw/rejected_2007_to_2018Q4.csv.gz     255 MB
下载方式见 README 的「数据获取」一节。

Windows 控制台是 GBK，全程不 print emoji，状态用 [OK] / [FAIL] 标记。
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import config as C          # noqa: E402

STEPS = [
    (1, "step1_prepare.py", "抽取建模全集 + 防泄漏字段过滤", [C.ACCEPTED_CSV]),
    (2, "step2_vintage_label.py", "vintage 定表现期 + 好坏客户定义", []),
    (3, "step3_scorecard.py", "跨时间切分 + 评分卡建模 + 三段验证", []),
    (4, "step4_benchmark.py", "对照模型（XGBoost / 加 LC 定价变量）", []),
    (5, "step5_reject_inference.py", "拒绝推断（重加权 + 打包法）", [C.REJECTED_CSV]),
    (6, "step6_strategy_profit.py", "策略表与收益测算", []),
    (7, "step7_monitoring.py", "上线后监控（设计 + 实跑）", []),
]


def check_data() -> bool:
    ok = True
    print("=" * 72)
    print("  数据就位检查")
    print("=" * 72)
    for p, size_mb, why in ((C.ACCEPTED_CSV, 1675, "主数据：已核准贷款"),
                            (C.REJECTED_CSV, 255, "拒绝推断用：被拒申请")):
        if p.exists():
            mb = p.stat().st_size / 1e6
            flag = "[OK]  " if abs(mb - size_mb) / size_mb < 0.02 else "[WARN]"
            print(f"{flag} {p.name:42s} {mb:8.1f} MB  （期望 {size_mb} MB）{why}")
            if flag.startswith("[WARN]"):
                print(f"       文件大小对不上，可能是下载不完整，建议重下")
        else:
            print(f"[FAIL] {p.name:42s} 缺失  —— {why}")
            print(f"       期望路径：{p}")
            ok = False
    if not ok:
        print("")
        print("数据获取方式见 README.md 的「数据获取」一节。")
    return ok


def run(step_file: str) -> int:
    return subprocess.call([sys.executable, str(ROOT / "steps" / step_file)])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--only", dest="only", type=int, default=None)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()

    if not check_data():
        return 1
    if a.check:
        return 0

    if a.only:
        todo = [s for s in STEPS if s[0] == a.only]
    else:
        todo = [s for s in STEPS if s[0] >= a.start]
    t0 = time.time()
    print("")
    for no, f, name, needs in todo:
        miss = [p for p in needs if not p.exists()]
        if miss:
            print(f"[SKIP] 第 {no} 步 {name} —— 缺少 {[p.name for p in miss]}")
            continue
        print("#" * 72)
        print(f"#  第 {no} 步  {name}")
        print("#" * 72)
        t = time.time()
        rc = run(f)
        if rc != 0:
            print(f"[FAIL] 第 {no} 步失败（退出码 {rc}），中止")
            return rc
        print(f"[OK] 第 {no} 步完成，用时 {time.time()-t:.0f}s")
        print("")

    print("=" * 72)
    print(f"[OK] 全流程完成，总用时 {time.time()-t0:.0f}s")
    print(f"     产物在 {C.OUT}")
    print(f"     日志在 {C.OUT}/step*_日志.txt")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
