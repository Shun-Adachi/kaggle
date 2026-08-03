# CV で正解率が上がらない難所層(male C1 / female C3 / male C3)について、
# 「モデルが当てられる行」と「外す行」のデータの性質を比較する。
# 行の正誤は repeated CV(seed 0〜9)の OOF 多数決で安定化し、実際の生死 × 予測で
# 4グループに分ける:
#   TP = 生存・生存と予測(当てた) / FN = 生存・死亡と予測(取りこぼし)
#   TN = 死亡・死亡と予測(当てた) / FP = 死亡・生存と予測(楽観)
# 「TP vs FN」「TN vs FP」の分布が重なっているほど、その層は属性から見分け不能(=運)。
# 出力: notebooks/figures/hard_segments/<日付_モデル_特徴セット>/ に層ごとの図 + group_stats.csv
# 実行(リポジトリルートから):
#   .venv/bin/python competitions/titanic/notebooks/hard_segment_eda.py [--model lgbm_t] [--features wcg2sc]
import argparse
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

OUTBASE = "competitions/titanic/notebooks/figures/hard_segments"
SEEDS = list(range(10))
SEGMENTS = ["male C1", "female C3", "male C3"]

# dataviz 参照パレット。色相=実際の生死(blue=生存 / orange=死亡)、
# 濃淡=予測の正誤(濃=当てた / 淡=外した)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_TP = "#2a78d6"  # 生存を当てた
C_FN = "#9ec5f4"  # 生存を外した(淡い blue)
C_TN = "#eb6834"  # 死亡を当てた
C_FP = "#f6c3a8"  # 死亡を外した(淡い orange)
GROUPS = [("TP", "survived, caught", C_TP), ("FN", "survived, missed", C_FN),
          ("TN", "died, caught", C_TN), ("FP", "died, missed", C_FP)]

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
        "font.size": 10,
    }
)


def style_ax(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)


def oof_stats(model: str, fs_name: str, train_df: pd.DataFrame) -> pd.DataFrame:
    """seed 0〜9 の OOF から行ごとの平均確率・多数決予測を求め、4グループに分類。"""
    probas = np.column_stack([run_cv(model, fs_name, train_df, seed=s) for s in SEEDS])
    mean_p = probas.mean(axis=1)
    vote = (probas >= 0.5).mean(axis=1)  # 生存と予測した seed の割合
    pred = (vote >= 0.5).astype(int)
    y = train_df["Survived"].to_numpy()
    grp = np.select(
        [(y == 1) & (pred == 1), (y == 1) & (pred == 0), (y == 0) & (pred == 0)],
        ["TP", "FN", "TN"], default="FP",
    )
    out = train_df.copy()
    out["MeanProba"], out["VoteShare"], out["Group"] = mean_p, vote, grp
    out["Seg"] = out["Sex"] + " C" + out["Pclass"].astype(str)
    return out


def box_with_points(ax, df, col, title, unit=""):
    """4グループの分布: 箱ひげ + 実点(ジッター)。点が主役、箱は補助。"""
    data, colors = [], []
    for g, _, c in GROUPS:
        data.append(df.loc[df["Group"] == g, col].dropna())
        colors.append(c)
    bp = ax.boxplot(data, positions=range(4), widths=0.5, patch_artist=True,
                    showfliers=False, medianprops={"color": INK, "linewidth": 1.4},
                    whiskerprops={"color": BASELINE}, capprops={"color": BASELINE})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_edgecolor(BASELINE)
        patch.set_alpha(0.5)
    rng = np.random.default_rng(0)
    for i, (vals, c) in enumerate(zip(data, colors)):
        ax.scatter(i + rng.uniform(-0.16, 0.16, len(vals)), vals, s=14, color=c,
                   edgecolor=SURFACE, linewidth=0.5, zorder=3)
    ax.set_xticks(range(4), [f"{g}\nn={len(d)}" for (g, _, _), d in zip(GROUPS, data)])
    ax.set_ylabel(unit or col)
    style_ax(ax, title)


def share_bars(ax, df, title):
    """4グループ別の「該当割合」: WCG証拠あり / Cabin記録あり / S港 / 同行者あり。"""
    feats = [
        ("WCG evidence", df["WcGroupInfo"] == 1),
        ("Cabin recorded", df["CabinFlag"] == 1),
        ("Embarked S", df["Embarked"] == "S"),
        ("group >= 2", df["TicketGroupSize"] >= 2),
    ]
    x = np.arange(len(feats))
    width = 0.19
    for j, (g, _, c) in enumerate(GROUPS):
        mask = df["Group"] == g
        shares = [flag[mask].mean() * 100 if mask.any() else 0 for _, flag in feats]
        ax.bar(x + (j - 1.5) * width, shares, width=width * 0.92, color=c,
               label=g, zorder=2)
    ax.set_xticks(x, [f for f, _ in feats])
    ax.set_ylabel("share (%)")
    ax.set_ylim(0, 105)
    ax.legend(frameon=False, labelcolor=INK2, fontsize=8, ncol=4)
    style_ax(ax, title)


def proba_hist(ax, df, title):
    """OOF 平均確率の分布: 当てた行(濃) vs 外した行(淡)。
    外した行が 0/1 の端に張り付くほど「自信満々の間違い」= 属性からは救えない。"""
    bins = np.arange(0, 1.05, 0.1)
    correct = df[df["Group"].isin(["TP", "TN"])]["MeanProba"]
    wrong = df[df["Group"].isin(["FN", "FP"])]["MeanProba"]
    ax.hist(correct, bins=bins, color=C_TP, alpha=0.45, label="correct rows", zorder=2)
    ax.hist(wrong, bins=bins, histtype="step", linewidth=2.2, color=C_TN,
            label="wrong rows", zorder=3)
    ax.axvline(0.5, color=BASELINE, lw=1.2, ls=":")
    ax.set_xlabel("mean OOF probability of survival (10 seeds)")
    ax.set_ylabel("passengers")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    style_ax(ax, title)


def segment_figure(df_seg: pd.DataFrame, seg: str, outdir: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    err = df_seg["Group"].isin(["FN", "FP"]).mean()
    proba_hist(axes[0, 0], df_seg, f"Model confidence: correct vs wrong rows")
    box_with_points(axes[0, 1], df_seg, "Age", "Age by outcome group", "age (years)")
    box_with_points(axes[1, 0], df_seg.assign(
        FarePerPerson=df_seg["FarePerPerson"].clip(upper=80)),
        "FarePerPerson", "Fare per person by outcome group", "fare/person (capped 80)")
    share_bars(axes[1, 1], df_seg, "Feature shares by outcome group")
    fig.suptitle(
        f"{seg} (n={len(df_seg)}, OOF error {err:.0%})  —  "
        f"TP/TN = caught, FN/FP = missed",
        color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    fig.savefig(f"{outdir}/{seg.replace(' ', '_')}.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lgbm_t")
    parser.add_argument("--features", default="wcg2sc", choices=list(FEATURE_SETS))
    args = parser.parse_args()

    train_df, _ = load_train_test()
    # 図で使う WCG 証拠の有無は「全 train を学習側にした場合」(test 予測時と同じ条件)
    fs = FEATURE_SETS[args.features]
    train_g = apply_group(fs, train_df, train_df)

    print(f"OOF を計算中(seed 0〜9, {args.model}/{args.features})...")
    df = oof_stats(args.model, args.features, train_df)
    df["WcGroupInfo"] = train_g["WcGroupInfo"]

    outdir = f"{OUTBASE}/{date.today():%Y%m%d}_{args.model}_{args.features}"
    os.makedirs(outdir, exist_ok=True)

    rows = []
    for seg in SEGMENTS:
        df_seg = df[df["Seg"] == seg]
        segment_figure(df_seg, seg, outdir)
        for g, desc, _ in GROUPS:
            d = df_seg[df_seg["Group"] == g]
            if len(d) == 0:
                continue
            rows.append({
                "segment": seg, "group": g, "desc": desc, "n": len(d),
                "age_median": d["Age"].median(),
                "fare_pp_median": round(d["FarePerPerson"].median(), 1),
                "wcg_evidence": round((d["WcGroupInfo"] == 1).mean(), 2),
                "cabin": round((d["CabinFlag"] == 1).mean(), 2),
                "embarked_S": round((d["Embarked"] == "S").mean(), 2),
                "mean_proba": round(d["MeanProba"].mean(), 2),
            })
    stats = pd.DataFrame(rows)
    stats.to_csv(f"{outdir}/group_stats.csv", index=False)
    print(stats.to_string(index=False))
    print(f"\n図とサマリを保存: {outdir}/")


if __name__ == "__main__":
    main()
