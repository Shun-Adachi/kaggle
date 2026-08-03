# Age 補完の精緻化の検討: 補完法ごとの「隠した Age を当てる」精度(CV MAE)と、
# Title×Pclass の年齢構造・欠損者の分布を可視化する。
# 結論(2026-08-03): MAE は 9.4→8.6 歳まで改善できるが、survival のペア比較 repeated CV では
# 全体 ±0.5人以内の同格(wcg2scp/wcg2scr、詳細はチケット log)。年齢は既に使い切られている。
# 出力: notebooks/figures/age_imputation/<日付>/ に図1枚 + summary.json
# 実行(リポジトリルートから): .venv/bin/python competitions/titanic/notebooks/age_imputation_eda.py
import json
import os
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.model_selection import KFold

sys.path.insert(0, "competitions/titanic/src")
from train import load_train_test  # noqa: E402

OUTBASE = "competitions/titanic/notebooks/figures/age_imputation"

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"
ORANGE = "#eb6834"
BLUES = {1: "#1c5cab", 2: "#3987e5", 3: "#9ec5f4"}  # Pclass 1/2/3(同一色相の濃淡)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.8, "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "axes.spines.left": False, "font.size": 11,
    }
)


def style_ax(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)


def imputers(known):
    def by_global(tr, va):
        return np.full(len(va), tr["Age"].median())

    def by_title(tr, va):
        med = tr.groupby("Title")["Age"].median()
        return va["Title"].map(med).fillna(tr["Age"].median()).to_numpy()

    def by_title_pclass(tr, va):
        med = tr.groupby(["Title", "Pclass"])["Age"].median()
        fb = tr.groupby("Title")["Age"].median()
        out = pd.Series([med.get(k, np.nan) for k in zip(va["Title"], va["Pclass"])],
                        index=va.index)
        return out.fillna(va["Title"].map(fb)).fillna(tr["Age"].median()).to_numpy()

    def by_regression(tr, va):
        cols = ["Pclass", "SibSp", "Parch", "Fare", "FamilySize", "TicketGroupSize", "FarePerPerson"]

        def enc(df):
            X = df[cols].copy()
            for c, vals in {"Title": ["Mr", "Miss", "Mrs", "Master", "Rare"],
                            "Sex": ["male", "female"], "Embarked": ["S", "C", "Q"]}.items():
                for v in vals:
                    X[f"{c}_{v}"] = (df[c] == v).astype(int)
            return X

        m = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                          random_state=42, verbose=-1)
        m.fit(enc(tr), tr["Age"])
        return m.predict(enc(va))

    return [("global median", by_global), ("Title median\n(current)", by_title),
            ("Title x Pclass\nmedian", by_title_pclass), ("LGBM\nregression", by_regression)]


def cv_mae(known, fn):
    errs = []
    for tr_idx, va_idx in KFold(5, shuffle=True, random_state=42).split(known):
        tr, va = known.iloc[tr_idx], known.iloc[va_idx]
        errs.append(np.abs(fn(tr, va) - va["Age"].to_numpy()).mean())
    return float(np.mean(errs))


def main() -> None:
    train_df, test_df = load_train_test()
    known = train_df[train_df["Age"].notna()].reset_index(drop=True)
    outdir = f"{OUTBASE}/{date.today():%Y%m%d}"
    os.makedirs(outdir, exist_ok=True)

    maes = [(name, cv_mae(known, fn)) for name, fn in imputers(known)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # 左: 補完法ごとの CV MAE
    ax = axes[0]
    names = [n for n, _ in maes]
    vals = [v for _, v in maes]
    colors = [BLUE if "current" not in n else ORANGE for n in names]
    ax.bar(range(len(maes)), vals, width=0.55, color=colors, zorder=2)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.2f}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(range(len(maes)), names)
    ax.set_ylabel("CV MAE of hidden Age (years)")
    style_ax(ax, "How well can missing Age be reconstructed?")

    # 右: Title×Pclass の年齢中央値(欠損者の落ち先が階級で大きく違う)
    ax = axes[1]
    med = train_df.pivot_table(index="Title", columns="Pclass", values="Age", aggfunc="median")
    miss = train_df[train_df["Age"].isna()].groupby(["Title", "Pclass"]).size().unstack(fill_value=0)
    titles = ["Mr", "Miss", "Mrs", "Master"]
    x = np.arange(len(titles))
    for j, pc in enumerate([1, 2, 3]):
        vals = [med.loc[t, pc] for t in titles]
        ax.bar(x + (j - 1) * 0.26, vals, width=0.24, color=BLUES[pc],
               label=f"class {pc}", zorder=2)
        for xi, t in zip(x, titles):
            n = miss.loc[t, pc] if t in miss.index and pc in miss.columns else 0
            if n:
                ax.text(xi + (j - 1) * 0.26, med.loc[t, pc] + 0.7, f"{n}",
                        ha="center", color=INK2, fontsize=9)
    ax.set_xticks(x, titles)
    ax.set_ylabel("median Age (years)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    style_ax(ax, "Median Age by Title x Pclass (labels = missing-Age count landing there)")

    fig.suptitle("Age imputation study — better MAE, but survival CV unchanged (see ticket log)",
                 color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(f"{outdir}/01_age_imputation.png", dpi=150)
    plt.close(fig)

    with open(f"{outdir}/summary.json", "w") as f:
        json.dump({"cv_mae": {n.replace('\n', ' '): round(v, 2) for n, v in maes},
                   "missing_train": int(train_df["Age"].isna().sum()),
                   "missing_test": int(test_df["Age"].isna().sum())}, f, indent=2)
    for n, v in maes:
        print(f"{n.replace(chr(10), ' '):24s} MAE {v:.2f}")
    print(f"図とサマリを保存: {outdir}/")


if __name__ == "__main__":
    main()
