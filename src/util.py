# -*- coding: utf-8 -*-
"""跑批日志与绘图的公共设置。Windows 控制台是 GBK，全程不 print emoji。"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


class Logger:
    """同时打到控制台和文件的极简日志。每一步的产物都要留下可追溯的日志。"""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.lines: list[str] = []

    def __call__(self, msg: str = "") -> None:
        print(msg)
        self.lines.append(str(msg))

    def section(self, title: str) -> None:
        self("")
        self("=" * 72)
        self(f"  {title}")
        self("=" * 72)

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text("\n".join(self.lines), encoding="utf-8")
        print(f"[OK] 日志 -> {self.path}")


def fmt_df(df, floatfmt: dict | None = None, max_rows: int = 60) -> str:
    """把 DataFrame 转成日志里好读的等宽文本。"""
    d = df.head(max_rows)
    return d.to_string(index=False, formatters=floatfmt or {})


def pct(x) -> str:
    return f"{x:.2%}"


def downcast(df, cat_max_card: int = 1000):
    """把 DataFrame 压到最小内存：float64 -> float32，低基数字符串 -> category。

    不是为了好看。这台机器上 1.43M x 95 的表按默认 dtype 反序列化会直接把
    pyarrow 打到段错误（exit 139），压完之后内存占用降到三分之一以下。
    评分卡全程只用到分箱后的 WOE，float32 的精度绰绰有余。
    """
    import numpy as np
    import pandas as pd

    out = df.copy()
    for c in out.columns:
        s = out[c]
        if pd.api.types.is_float_dtype(s) and s.dtype != np.float32:
            out[c] = s.astype(np.float32)
        elif pd.api.types.is_integer_dtype(s):
            out[c] = pd.to_numeric(s, downcast="integer")
        elif s.dtype == object or pd.api.types.is_string_dtype(s):
            if s.nunique(dropna=True) <= cat_max_card:
                out[c] = s.astype("category")
    return out
