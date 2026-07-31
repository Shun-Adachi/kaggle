# 誤答分析: CV の out-of-fold 予測(各乗客を、その人を学習に使っていないモデルで予測)を使い、
# モデルがどのセグメントで間違えているかを俯瞰する。
# 出力: notebooks/figures/error_analysis/<日付_モデル_特徴セット_cvスコア>/ に
#       図・誤答CSV・セグメント集計・実行条件(run_info.json)をセットで保存。
#       条件が違えば別ディレクトリになるため、パラメータを変えても過去の結果は消えない。
#       ディレクトリは git 管理外(.gitignore)。
# 実行(リポジトリルートから): .venv/bin/python competitions/titanic/notebooks/error_analysis.py [--model lgbm] [--features full]
import argparse
import json
import os
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, "competitions/titanic/src")
from train import FEATURE_SETS, load_train_test, run_cv  # noqa: E402

OUTBASE = "competitions/titanic/notebooks/figures/error_analysis"

# dataviz 参照パレット(rulebase_eda.py と共通)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_SURVIVED = "#2a78d6"  # blue: 実際は生存(FN: 生存者を死亡と予測)
C_DIED = "#eb6834"  # orange: 実際は死亡(FP: 死亡者を生存と予測)

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "text.color": INK,
        "axes.edgecolor": BASELINE,
        "axes.labelcolor": INK2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "font.size": 11,
    }
)


def style_ax(ax, title):
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)


def error_bar(ax, labels, rates, counts, title, color=C_DIED):
    """誤答率の棒グラフ(単系列)。棒上にレート、x ラベルに n。"""
    x = np.arange(len(labels))
    ax.bar(x, rates, width=0.6, color=color, zorder=2)
    for xi, r in zip(x, rates):
        ax.text(xi, r + 0.008, f"{r:.0%}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, [f"{l}\nn={n}" for l, n in zip(labels, counts)])
    ax.set_ylim(0, max(rates.max() * 1.3, 0.05))
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    style_ax(ax, title)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm",
                        choices=["logreg", "rf", "lgbm", "lgbm_t", "ens"])
    parser.add_argument("--features", default="full", choices=list(FEATURE_SETS))
    args = parser.parse_args()

    df, _ = load_train_test()
    fs = FEATURE_SETS[args.features]
    df["proba"] = run_cv(args.model, args.features, df)
    df["pred"] = (df["proba"] >= 0.5).astype(int)
    df["correct"] = df["pred"] == df["Survived"]
    acc = df["correct"].mean()
    n_fn = int(((df.Survived == 1) & ~df.correct).sum())
    n_fp = int(((df.Survived == 0) & ~df.correct).sum())
    print(f"=== OOF 全体 ({args.model} / {args.features}) ===")
    print(f"accuracy: {acc:.4f}  誤答 {(~df['correct']).sum()}/{len(df)} 件")
    print(f"  FN(生存者を死亡と予測): {n_fn} 件")
    print(f"  FP(死亡者を生存と予測): {n_fp} 件")

    # 実行条件+スコアごとの保存先(条件を変えても過去の結果が消えない)
    outdir = f"{OUTBASE}/{date.today():%Y%m%d}_{args.model}_{args.features}_cv{acc:.4f}"
    os.makedirs(outdir, exist_ok=True)
    run_info = {
        "date": f"{date.today():%Y-%m-%d}",
        "model": args.model,
        "features": args.features,
        "feature_cols": fs["num"] + fs["cat"],
        "cv": "StratifiedKFold 5-fold, shuffle=True, seed=42",
        "metric": "accuracy",
        "oof_accuracy": round(float(acc), 4),
        "n_errors": int((~df["correct"]).sum()),
        "n_fn": n_fn,
        "n_fp": n_fp,
        "script": "notebooks/error_analysis.py",
    }
    with open(f"{outdir}/run_info.json", "w") as f:
        json.dump(run_info, f, ensure_ascii=False, indent=2)

    # --- 1. Sex × Pclass: 誤答率と誤答の向き ---
    df["Segment"] = df["Sex"] + " C" + df["Pclass"].astype(str)
    seg_order = [f"{s} C{c}" for s in ["female", "male"] for c in [1, 2, 3]]
    g = df.groupby("Segment").agg(
        n=("correct", "size"),
        err=("correct", lambda s: (~s).sum()),
        err_rate=("correct", lambda s: (~s).mean()),
        fn=("correct", "size"),  # 後で上書き
    ).loc[seg_order]
    g["fn"] = df[(df.Survived == 1) & ~df.correct].groupby("Segment").size().reindex(seg_order, fill_value=0)
    g["fp"] = df[(df.Survived == 0) & ~df.correct].groupby("Segment").size().reindex(seg_order, fill_value=0)
    print("\n=== Sex × Pclass 別 誤答 ===")
    print(g[["n", "err", "err_rate", "fn", "fp"]].round(3))
    g[["n", "err", "err_rate", "fn", "fp"]].round(4).to_csv(f"{outdir}/segment_errors.csv")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    error_bar(axes[0], seg_order, g["err_rate"], g["n"], "OOF error rate by Sex × Pclass")
    axes[0].tick_params(axis="x", labelsize=9)
    ax = axes[1]
    x = np.arange(len(seg_order))
    ax.bar(x, g["fn"], 0.6, color=C_SURVIVED, label="FN: survivor predicted dead",
           zorder=2, edgecolor=SURFACE, linewidth=1.5)
    ax.bar(x, g["fp"], 0.6, bottom=g["fn"], color=C_DIED, label="FP: victim predicted alive",
           zorder=2, edgecolor=SURFACE, linewidth=1.5)
    for xi, (fn, fp) in enumerate(zip(g["fn"], g["fp"])):
        ax.text(xi, fn + fp + 0.6, str(fn + fp), ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, seg_order)
    ax.tick_params(axis="x", labelsize=9)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    style_ax(ax, "Error counts & direction by segment")
    fig.savefig(f"{outdir}/01_sex_pclass.png", dpi=150)

    # --- 2. その他の切り口: Title / TicketGroupSize / Age帯 / Embarked ---
    df["AgeBand"] = pd.cut(df["Age"], [0, 10, 18, 30, 40, 50, 80],
                           labels=["0-10", "10-18", "18-30", "30-40", "40-50", "50-80"])
    df["AgeBand"] = df["AgeBand"].cat.add_categories("missing").fillna("missing")
    df["TicketGroupBand"] = df["TicketGroupSize"].clip(upper=5).astype(int).astype(str).replace("5", "5+")
    dims = [("Title", ["Mr", "Miss", "Mrs", "Master", "Rare"]),
            ("TicketGroupBand", ["1", "2", "3", "4", "5+"]),
            ("AgeBand", None), ("Embarked", ["C", "Q", "S"])]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), layout="constrained")
    for ax, (col, order) in zip(axes.flat, dims):
        gg = df.groupby(col, observed=True)["correct"].agg(n="size", err_rate=lambda s: (~s).mean())
        if order:
            gg = gg.loc[[o for o in order if o in gg.index]]
        error_bar(ax, list(gg.index), gg["err_rate"], gg["n"], f"OOF error rate by {col}")
        ax.tick_params(axis="x", labelsize=9)
        print(f"\n=== {col} 別 誤答率 ===\n", gg.round(3))
    fig.savefig(f"{outdir}/02_other_dims.png", dpi=150)

    # --- 3. 予測確率の分布(自信を持って間違えたか) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    bins = np.linspace(0, 1, 21)
    ax = axes[0]
    ax.hist([df.loc[df.correct, "proba"], df.loc[~df.correct, "proba"]], bins=bins,
            color=[MUTED, C_DIED], label=["correct", "wrong"], zorder=2, rwidth=0.9)
    ax.axvline(0.5, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xlabel("OOF predicted survival probability")
    ax.legend(frameon=False, labelcolor=INK2)
    style_ax(ax, "Prediction confidence: correct vs wrong")
    ax = axes[1]
    wrong = df[~df.correct]
    ax.hist([wrong.loc[wrong.Survived == 1, "proba"], wrong.loc[wrong.Survived == 0, "proba"]],
            bins=bins, color=[C_SURVIVED, C_DIED],
            label=["FN (actually survived)", "FP (actually died)"], zorder=2, rwidth=0.9)
    ax.axvline(0.5, color=BASELINE, linewidth=1, zorder=1)
    ax.set_xlabel("OOF predicted survival probability")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    style_ax(ax, "Errors only: how confidently wrong?")
    fig.savefig(f"{outdir}/03_confidence.png", dpi=150)

    # --- 4. 誤答一覧を CSV に保存 + 自信満々の誤答トップを表示 ---
    cols = ["PassengerId", "Name", "Sex", "Pclass", "Age", "Title", "FamilySize",
            "TicketGroupSize", "Ticket", "Fare", "Embarked", "Cabin",
            "Survived", "pred", "proba"]
    errors = df.loc[~df.correct, cols].copy()
    errors["confidence"] = (errors["proba"] - 0.5).abs()
    errors = errors.sort_values(["Sex", "Pclass", "confidence"], ascending=[True, True, False])
    csv_path = f"{outdir}/errors_oof.csv"
    errors.round({"proba": 3, "confidence": 3}).to_csv(csv_path, index=False)
    print(f"\n誤答 {len(errors)} 件を保存: {csv_path}")
    print("\n=== 自信を持って間違えたトップ10 ===")
    top = errors.sort_values("confidence", ascending=False).head(10)
    print(top[["Name", "Sex", "Pclass", "Age", "TicketGroupSize", "Survived", "proba"]].to_string(index=False))


if __name__ == "__main__":
    main()
