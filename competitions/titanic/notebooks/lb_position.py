# 公開リーダーボード全体の分布の中で、自分の提出がどこに立っているかを可視化する。
# LB スナップショットは kaggle CLI で取得して data/leaderboard/ に置く(git 管理外):
#   .venv/bin/kaggle competitions leaderboard titanic --download -p competitions/titanic/data/leaderboard
#   unzip -o competitions/titanic/data/leaderboard/titanic.zip -d competitions/titanic/data/leaderboard
# 出力: notebooks/figures/lb_position/<スナップショット日付>/ に図2枚 + summary.json。
# 実行(リポジトリルートから): .venv/bin/python competitions/titanic/notebooks/lb_position.py
import glob
import json
import os
import re

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COMP_DIR = "competitions/titanic"
OUTBASE = f"{COMP_DIR}/notebooks/figures/lb_position"

# 1問の重み。test は 418 人なので 1 人当てるごとに 1/418 = 0.239% 上がる
STEP = 1 / 418

# 自分の提出履歴(README の結果表と同期して手で更新する)
MY_SUBS = {
    "baseline logreg (07-29)": 0.76794,
    "best: lgbm_t/wcg2sc (07-31)": 0.77751,
}
MY_BEST = max(MY_SUBS.values())

# 参照点
REFS = {
    "all dead": 0.62200,
    "gender rule": 0.76555,  # 公式サンプル(女性=生存)の既知スコア
}

# dataviz 参照パレット(error_analysis.py と共通)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
BLUE = "#2a78d6"   # 分布(単一系列)
ORANGE = "#eb6834"  # 自分の位置(強調)

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


def load_lb() -> tuple[pd.Series, str]:
    paths = sorted(glob.glob(f"{COMP_DIR}/data/leaderboard/titanic-publicleaderboard-*.csv"))
    if not paths:
        raise SystemExit("LB スナップショットがない。ファイル冒頭のコマンドで取得すること。")
    path = paths[-1]  # 最新スナップショット
    snap_date = re.search(r"(\d{4}-\d{2}-\d{2})", os.path.basename(path)).group(1)
    return pd.read_csv(path)["Score"], snap_date


def top_pct(scores: pd.Series, v: float) -> float:
    """スコア v の上位パーセント(小さいほど上位)。"""
    return (scores > v).mean() * 100


def fig_distribution(scores: pd.Series, outdir: str) -> None:
    """LB スコア分布(0.60〜0.85)と自分の位置。"""
    lo, hi = 0.60, 0.85
    in_range = scores[(scores >= lo) & (scores <= hi)]
    # ビン幅 = 2問分。ビン境界を 1/418 の格子に合わせて同点クラスタが割れないようにする
    edges = np.arange(round(lo / STEP), round(hi / STEP) + 2, 2) * STEP - STEP / 2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(in_range, bins=edges, color=BLUE, zorder=2)
    ax.set_xlim(lo, hi)

    def mark(v, label, color, ls, ytext, side="right"):
        ax.axvline(v, color=color, lw=1.6 if color == ORANGE else 1.2, ls=ls, zorder=3)
        dx = 0.0012 if side == "right" else -0.0012
        ax.text(v + dx, ytext, f"{label}\n{v:.5f} (top {top_pct(scores, v):.0f}%)",
                color=color if color == ORANGE else INK2, fontsize=9, va="top",
                ha="left" if side == "right" else "right")

    # 密集地帯(0.765〜0.778)はラベルを左右・高さで散らして重なりを避ける
    ymax = ax.get_ylim()[1]
    mark(REFS["all dead"], "all dead", MUTED, ":", ymax * 0.97)
    mark(REFS["gender rule"], "gender rule", MUTED, ":", ymax * 0.97, side="left")
    mark(MY_SUBS["baseline logreg (07-29)"], "our baseline", INK2, "--", ymax * 0.80, side="left")
    mark(scores.median(), "median", MUTED, ":", ymax * 0.63, side="left")
    mark(MY_BEST, "our best", ORANGE, "-", ymax * 0.97)

    n_out = (scores > hi).sum()
    ax.set_title(
        f"Public LB distribution ({len(scores):,} teams)  —  "
        f"{n_out} teams above {hi} (answer lookup) not shown",
        color=INK, fontsize=12, loc="left", pad=10,
    )
    ax.set_xlabel("Public LB score (1 passenger = 0.00239)")
    ax.set_ylabel("teams")
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(f"{outdir}/01_lb_distribution.png", dpi=150)
    plt.close(fig)


def fig_ladder(scores: pd.Series, outdir: str) -> None:
    """自己ベストから +1 人正解するごとに順位がどう変わるかの階段。"""
    steps = np.arange(0, 11)
    values = MY_BEST + steps * STEP
    pcts = [top_pct(scores, v) for v in values]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y = np.arange(len(steps))[::-1]
    bars = ax.barh(y, pcts, height=0.62, color=BLUE, zorder=2)
    bars[0].set_color(ORANGE)
    for yi, p, v, s in zip(y, pcts, values, steps):
        label = "current" if s == 0 else f"+{s}"
        ax.text(-0.6, yi, f"{label}  {v:.5f}", ha="right", va="center",
                color=ORANGE if s == 0 else INK2, fontsize=10)
        ax.text(p + 0.4, yi, f"top {p:.1f}%", va="center", color=INK, fontsize=10)
    ax.set_yticks([])
    ax.set_xlim(0, max(pcts) * 1.18)
    ax.set_xlabel("share of teams above this score (smaller = higher rank)")
    ax.set_title(
        "What one more correct passenger is worth (from our best 0.77751)",
        color=INK, fontsize=12, loc="left", pad=10,
    )
    ax.tick_params(length=0)
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.24)
    fig.savefig(f"{outdir}/02_score_ladder.png", dpi=150)
    plt.close(fig)


def main() -> None:
    scores, snap_date = load_lb()
    outdir = f"{OUTBASE}/{snap_date.replace('-', '')}"
    os.makedirs(outdir, exist_ok=True)

    summary = {
        "snapshot": snap_date,
        "teams": int(len(scores)),
        "median": float(scores.median()),
        "our_best": MY_BEST,
        "our_best_top_pct": round(top_pct(scores, MY_BEST), 1),
        "teams_above_085": int((scores > 0.85).sum()),
        "refs_top_pct": {k: round(top_pct(scores, v), 1) for k, v in {**REFS, **MY_SUBS}.items()},
        "ladder": {
            f"+{s}": {"score": round(MY_BEST + s * STEP, 5),
                      "top_pct": round(top_pct(scores, MY_BEST + s * STEP), 1)}
            for s in range(11)
        },
    }
    with open(f"{outdir}/summary.json", "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    fig_distribution(scores, outdir)
    fig_ladder(scores, outdir)

    print(f"snapshot {snap_date}: {len(scores):,} チーム")
    print(f"自己ベスト {MY_BEST} は上位 {summary['our_best_top_pct']}%(中央値 {summary['median']:.5f})")
    for s in (1, 2, 5, 10):
        d = summary["ladder"][f"+{s}"]
        print(f"  +{s}人正解 → {d['score']} (top {d['top_pct']}%)")
    print(f"図とサマリを保存: {outdir}/")


if __name__ == "__main__":
    main()
