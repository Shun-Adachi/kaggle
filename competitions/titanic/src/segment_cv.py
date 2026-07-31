"""セグメント別 OOF 正解率を repeated CV(seed 0〜9)で比較する評価器。

実行(リポジトリルートから):
    .venv/bin/python competitions/titanic/src/segment_cv.py --configs lgbm_t/wcg2 lgbm_t/wcg2c

- 各構成(モデル/特徴セット)を同じ10 seedで OOF 評価し、Sex×Pclass セグメント別に
  正解率の平均±stdを表示する。
- 先頭の構成を基準とし、他構成には差分と「2σ判定」(基準のσ×2 を超えた差か)を付ける。
  2σ未満の差は分割運と区別できない=「同格」とみなす(README の実験ルール)。
"""

import argparse

import numpy as np
import pandas as pd

from train import load_train_test, run_cv

SEEDS = list(range(10))


def segment_table(model: str, fs: str, train_df: pd.DataFrame) -> pd.DataFrame:
    """seed ごとのセグメント別正解率(行=seed, 列=セグメント+overall)。"""
    seg = (train_df["Sex"] + " C" + train_df["Pclass"].astype(str)).rename("Segment")
    y = train_df["Survived"].to_numpy()
    rows = []
    for seed in SEEDS:
        correct = ((run_cv(model, fs, train_df, seed=seed) >= 0.5).astype(int) == y)
        r = pd.Series(correct).groupby(seg.values).mean()
        r["overall"] = correct.mean()
        rows.append(r)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--configs", nargs="+", default=["lgbm_t/wcg2"],
                        help="モデル/特徴セット の列挙。先頭が比較基準")
    args = parser.parse_args()

    train_df, _ = load_train_test()
    seg_order = [f"{s} C{c}" for s in ["female", "male"] for c in [1, 2, 3]] + ["overall"]
    n_by_seg = (train_df["Sex"] + " C" + train_df["Pclass"].astype(str)).value_counts()

    stats = {}
    for cfg in args.configs:
        model, fs = cfg.split("/")
        t = segment_table(model, fs, train_df)
        stats[cfg] = (t.mean(), t.std())
        print(f"done: {cfg}")

    base_cfg = args.configs[0]
    base_mean, base_std = stats[base_cfg]
    print(f"\n{'セグメント':12s} {'n':>5s}  " + "  ".join(f"{c:>22s}" for c in args.configs)
          + "   (基準との差 / 2σ判定)")
    for s in seg_order:
        n = len(train_df) if s == "overall" else n_by_seg[s]
        cells = []
        for cfg in args.configs:
            mean, std = stats[cfg]
            cell = f"{mean[s]:.3f} ± {std[s]:.3f}"
            if cfg != base_cfg:
                diff = mean[s] - base_mean[s]
                sig = "★有意" if abs(diff) > 2 * base_std[s] else "同格"
                cell += f" ({diff:+.3f} {sig})"
            cells.append(f"{cell:>22s}")
        print(f"{s:12s} {n:>5d}  " + "  ".join(cells))


if __name__ == "__main__":
    main()
