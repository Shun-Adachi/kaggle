# test 側の改善余地の分析: LB の残り誤答(418 × (1-0.77751) ≈ 93人)がどのセグメントに
# 潜んでいるかを、OOF 誤答率(repeated CV: seed 0〜9 平均)× test の人数構成から推定する。
# あわせて WCG 証拠(同行者の生死情報)が test で届く範囲と、gender rule からの差分も可視化する。
# 出力: notebooks/figures/test_gap/<日付_モデル_特徴セット>/ に図3枚 + summary.json。
# 実行(リポジトリルートから):
#   .venv/bin/python competitions/titanic/notebooks/test_gap_analysis.py [--model lgbm_t] [--features wcg2sc]
import argparse
import glob
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
from train import FEATURE_SETS, apply_group, load_train_test, run_cv  # noqa: E402

OUTBASE = "competitions/titanic/notebooks/figures/test_gap"
SEEDS = list(range(10))  # repeated CV と同じ(選択に使った seed42 は除外)

# dataviz 参照パレット(error_analysis.py と共通)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_FN = "#2a78d6"   # blue: FN(生存者を死亡と予測)
C_FP = "#eb6834"   # orange: FP(死亡者を生存と予測)
C_COV = "#2a78d6"  # blue: WCG 証拠あり
C_NOCOV = "#c3c2b7"  # gray: 証拠なし

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

SEG_ORDER = [f"{s} C{c}" for s in ["female", "male"] for c in [1, 2, 3]]


def style_ax(ax, title):
    ax.set_title(title, color=INK, fontsize=12, loc="left", pad=10)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)


def segment(df: pd.DataFrame) -> pd.Series:
    return df["Sex"] + " C" + df["Pclass"].astype(str)


def oof_error_probs(model: str, fs: str, train_df: pd.DataFrame) -> pd.DataFrame:
    """seed 0〜9 の OOF から、乗客ごとの誤答確率(FN/FP 別)を推定する。"""
    y = train_df["Survived"].to_numpy()
    fn = np.zeros(len(train_df))
    fp = np.zeros(len(train_df))
    for s in SEEDS:
        pred = (run_cv(model, fs, train_df, seed=s) >= 0.5).astype(int)
        fn += (pred == 0) & (y == 1)
        fp += (pred == 1) & (y == 0)
    return pd.DataFrame(
        {"seg": segment(train_df), "fn": fn / len(SEEDS), "fp": fp / len(SEEDS)}
    )


def fig_expected_errors(err_df: pd.DataFrame, test_df: pd.DataFrame,
                        lb_errors: float, outdir: str) -> pd.DataFrame:
    """OOF 誤答率 × test 人数 = セグメント別の期待誤答数(FN/FP 積み上げ)。"""
    rates = err_df.groupby("seg")[["fn", "fp"]].mean().reindex(SEG_ORDER)
    n_test = segment(test_df).value_counts().reindex(SEG_ORDER)
    exp_fn = rates["fn"] * n_test
    exp_fp = rates["fp"] * n_test

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(SEG_ORDER))
    ax.bar(x, exp_fn, width=0.6, color=C_FN, label="expected FN (survivor pred. dead)",
           zorder=2)
    ax.bar(x, exp_fp, bottom=exp_fn, width=0.6, color=C_FP,
           label="expected FP (victim pred. alive)", zorder=2,
           edgecolor=SURFACE, linewidth=2)
    for xi, (a, b) in enumerate(zip(exp_fn, exp_fp)):
        ax.text(xi, a + b + 0.4, f"{a + b:.0f}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, [f"{s}\nn={n}" for s, n in zip(SEG_ORDER, n_test)])
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    style_ax(
        ax,
        f"Estimated test errors by segment (OOF rate x test count)  —  "
        f"total {exp_fn.sum() + exp_fp.sum():.0f} vs LB actual ~{lb_errors:.0f}",
    )
    ax.set_ylabel("expected wrong predictions (passengers)")
    fig.tight_layout()
    fig.savefig(f"{outdir}/01_expected_test_errors.png", dpi=150)
    plt.close(fig)
    return pd.DataFrame({"n_test": n_test, "exp_fn": exp_fn, "exp_fp": exp_fp})


def fig_wcg_coverage(test_g: pd.DataFrame, outdir: str) -> pd.Series:
    """WCG 証拠(同行者の生死情報)が test で届く人数のセグメント別内訳。"""
    seg = segment(test_g)
    cov = test_g["WcGroupInfo"] == 1
    n_cov = seg[cov].value_counts().reindex(SEG_ORDER).fillna(0)
    n_all = seg.value_counts().reindex(SEG_ORDER)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(SEG_ORDER))
    ax.bar(x, n_cov, width=0.6, color=C_COV, label="WCG evidence reaches", zorder=2)
    ax.bar(x, n_all - n_cov, bottom=n_cov, width=0.6, color=C_NOCOV,
           label="no evidence (attributes only)", zorder=2,
           edgecolor=SURFACE, linewidth=2)
    for xi, (c, n) in enumerate(zip(n_cov, n_all)):
        ax.text(xi, n + 2, f"{c:.0f}/{n}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, SEG_ORDER)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    style_ax(
        ax,
        f"Group-fate evidence coverage on test  —  "
        f"{int(n_cov.sum())}/{int(n_all.sum())} passengers ({n_cov.sum() / n_all.sum():.0%})",
    )
    ax.set_ylabel("test passengers")
    fig.tight_layout()
    fig.savefig(f"{outdir}/02_wcg_coverage.png", dpi=150)
    plt.close(fig)
    return n_cov


def fig_vs_gender(sub: pd.DataFrame, test_df: pd.DataFrame, outdir: str) -> pd.DataFrame:
    """提出予測が gender rule(女性=生存)から動かした人数のセグメント別内訳。"""
    m = test_df[["PassengerId", "Sex", "Pclass"]].merge(sub, on="PassengerId")
    m["seg"] = segment(m)
    m["gender"] = (m["Sex"] == "female").astype(int)
    to_dead = m[(m["gender"] == 1) & (m["Survived"] == 0)]["seg"].value_counts()
    to_alive = m[(m["gender"] == 0) & (m["Survived"] == 1)]["seg"].value_counts()
    flips = pd.DataFrame({
        "female -> dead": to_dead.reindex(SEG_ORDER).fillna(0),
        "male -> alive": to_alive.reindex(SEG_ORDER).fillna(0),
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(SEG_ORDER))
    ax.bar(x - 0.17, flips["female -> dead"], width=0.3, color=C_FP,
           label="female flipped to dead", zorder=2)
    ax.bar(x + 0.17, flips["male -> alive"], width=0.3, color=C_FN,
           label="male flipped to alive", zorder=2)
    for xi, (a, b) in enumerate(zip(flips["female -> dead"], flips["male -> alive"])):
        if a:
            ax.text(xi - 0.17, a + 0.15, f"{a:.0f}", ha="center", color=INK, fontsize=10)
        if b:
            ax.text(xi + 0.17, b + 0.15, f"{b:.0f}", ha="center", color=INK, fontsize=10)
    total = int(flips.to_numpy().sum())
    ax.set_xticks(x, SEG_ORDER)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    style_ax(ax, f"Where our submission departs from the gender rule  —  {total} flips")
    ax.set_ylabel("test passengers")
    fig.tight_layout()
    fig.savefig(f"{outdir}/03_vs_gender_rule.png", dpi=150)
    plt.close(fig)
    return flips


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm_t")
    parser.add_argument("--features", default="wcg2sc", choices=list(FEATURE_SETS))
    parser.add_argument("--lb-score", type=float, default=0.77751,
                        help="この構成の実測 LB(期待誤答数との突き合わせ用)")
    args = parser.parse_args()

    train_df, test_df = load_train_test()
    outdir = f"{OUTBASE}/{date.today():%Y%m%d}_{args.model}_{args.features}"
    os.makedirs(outdir, exist_ok=True)

    print(f"OOF 誤答率を推定中(seed {SEEDS[0]}〜{SEEDS[-1]}, {args.model}/{args.features})...")
    err_df = oof_error_probs(args.model, args.features, train_df)
    lb_errors = (1 - args.lb_score) * len(test_df)
    exp = fig_expected_errors(err_df, test_df, lb_errors, outdir)

    fs = FEATURE_SETS[args.features]
    test_g = apply_group(fs, train_df, test_df)
    n_cov = fig_wcg_coverage(test_g, outdir)

    subs = sorted(glob.glob(f"competitions/titanic/submissions/*{args.model}_{args.features}_*.csv"))
    flips = None
    if subs:
        sub = pd.read_csv(subs[-1])
        flips = fig_vs_gender(sub, test_df, outdir)
        print(f"提出ファイル: {os.path.basename(subs[-1])}")

    summary = {
        "model": args.model, "features": args.features, "lb_score": args.lb_score,
        "lb_errors_actual": round(lb_errors, 1),
        "expected_errors_total": round(float((exp["exp_fn"] + exp["exp_fp"]).sum()), 1),
        "segments": {
            s: {
                "n_test": int(exp.loc[s, "n_test"]),
                "expected_errors": round(float(exp.loc[s, "exp_fn"] + exp.loc[s, "exp_fp"]), 1),
                "wcg_covered": int(n_cov[s]),
            }
            for s in SEG_ORDER
        },
    }
    if flips is not None:
        summary["flips_vs_gender"] = {
            k: {s: int(v) for s, v in col.items() if v}
            for k, col in flips.items()
        }
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n=== セグメント別 期待誤答数(test) ===")
    print(exp.assign(total=lambda d: d["exp_fn"] + d["exp_fp"]).round(1))
    print(f"\n合計 {summary['expected_errors_total']}(LB 実測 ~{lb_errors:.0f})")
    print(f"図とサマリを保存: {outdir}/")


if __name__ == "__main__":
    main()
