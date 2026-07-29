# ルールベース分析用 EDA: 生存/非生存と各要素の関係をグラフ化する
# 出力: notebooks/figures/rulebase/*.png と集計値の標準出力
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

DATA = "competitions/titanic/data/train.csv"
OUTDIR = "competitions/titanic/notebooks/figures/rulebase"

# dataviz 参照パレット(light mode)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_SURVIVED = "#2a78d6"  # slot1 blue
C_DIED = "#eb6834"  # slot2 orange

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


def rate_bar(ax, labels, rates, counts, title, xlabel=""):
    """生存率の棒グラフ(単系列=blue)。棒上にレートと n を直接ラベル。"""
    x = np.arange(len(labels))
    ax.bar(x, rates, width=0.6, color=C_SURVIVED, zorder=2)
    for xi, r in zip(x, rates):
        ax.text(xi, r + 0.02, f"{r:.0%}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, [f"{l}\nn={n}" for l, n in zip(labels, counts)])
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel(xlabel)
    style_ax(ax, title)


def main():
    import os

    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_csv(DATA)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    print("=== 基本情報 ===")
    print(f"rows={len(df)}  survived={df.Survived.mean():.3f}")
    print("欠損:", df.isna().sum()[lambda s: s > 0].to_dict())

    # --- 1. 全体 + 性別 ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    counts = df["Survived"].value_counts()
    axes[0].bar(
        ["Died", "Survived"],
        [counts[0], counts[1]],
        width=0.55,
        color=[C_DIED, C_SURVIVED],
        zorder=2,
    )
    for i, v in enumerate([counts[0], counts[1]]):
        axes[0].text(i, v + 8, str(v), ha="center", color=INK, fontsize=10)
    style_ax(axes[0], f"Overall: {df.Survived.mean():.0%} survived (n={len(df)})")
    g = df.groupby("Sex")["Survived"].agg(["mean", "size"]).loc[["female", "male"]]
    rate_bar(axes[1], ["female", "male"], g["mean"], g["size"], "Survival rate by Sex")
    fig.savefig(f"{OUTDIR}/01_overall_sex.png", dpi=150)
    print("\n=== Sex ===\n", g)

    # --- 2. Pclass / Sex×Pclass ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    g = df.groupby("Pclass")["Survived"].agg(["mean", "size"])
    rate_bar(
        axes[0], [f"Class {c}" for c in g.index], g["mean"], g["size"],
        "Survival rate by Pclass",
    )
    piv = df.pivot_table(index="Pclass", columns="Sex", values="Survived", aggfunc="mean")
    n_piv = df.pivot_table(index="Pclass", columns="Sex", values="Survived", aggfunc="size")
    x = np.arange(3)
    w = 0.35
    ax = axes[1]
    ax.bar(x - w / 2, piv["female"], w - 0.02, color=C_SURVIVED, label="female", zorder=2)
    ax.bar(x + w / 2, piv["male"], w - 0.02, color=C_DIED, label="male", zorder=2)
    for xi, (f, m, nf, nm) in enumerate(
        zip(piv["female"], piv["male"], n_piv["female"], n_piv["male"])
    ):
        ax.text(xi - w / 2, f + 0.02, f"{f:.0%}", ha="center", color=INK, fontsize=9)
        ax.text(xi + w / 2, m + 0.02, f"{m:.0%}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, [f"Class {c}" for c in piv.index])
    ax.set_ylim(0, 1.1)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, labelcolor=INK2)
    style_ax(ax, "Survival rate by Sex × Pclass")
    fig.savefig(f"{OUTDIR}/02_pclass.png", dpi=150)
    print("\n=== Sex × Pclass ===\n", piv.round(3), "\n", n_piv)

    # --- 3. Age ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ax = axes[0]
    bins = np.arange(0, 85, 5)
    a = df.dropna(subset=["Age"])
    ax.hist(
        [a.loc[a.Survived == 1, "Age"], a.loc[a.Survived == 0, "Age"]],
        bins=bins,
        color=[C_SURVIVED, C_DIED],
        label=["Survived", "Died"],
        zorder=2,
        rwidth=0.92,
        stacked=False,
    )
    ax.legend(frameon=False, labelcolor=INK2)
    ax.set_xlabel("Age")
    style_ax(ax, f"Age distribution (missing: {df.Age.isna().sum()})")
    band = pd.cut(a["Age"], [0, 10, 18, 30, 40, 50, 60, 80])
    g = a.groupby(band, observed=True)["Survived"].agg(["mean", "size"])
    rate_bar(
        axes[1],
        ["0-10", "10-18", "18-30", "30-40", "40-50", "50-60", "60-80"],
        g["mean"], g["size"], "Survival rate by age band",
    )
    axes[1].tick_params(axis="x", labelsize=9)
    fig.savefig(f"{OUTDIR}/03_age.png", dpi=150)
    print("\n=== Age band ===\n", g.round(3))

    # --- 4. Fare ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), layout="constrained")
    ax = axes[0]
    fb = np.arange(0, 110, 5)
    ax.hist(
        [df.loc[df.Survived == 1, "Fare"].clip(upper=105),
         df.loc[df.Survived == 0, "Fare"].clip(upper=105)],
        bins=fb, color=[C_SURVIVED, C_DIED], label=["Survived", "Died"],
        zorder=2, rwidth=0.92,
    )
    ax.legend(frameon=False, labelcolor=INK2)
    ax.set_xlabel("Fare (clipped at 105)")
    style_ax(ax, "Fare distribution")
    qb = pd.qcut(df["Fare"], 4, labels=["Q1 (low)", "Q2", "Q3", "Q4 (high)"])
    g = df.groupby(qb, observed=True)["Survived"].agg(["mean", "size"])
    rate_bar(axes[1], list(g.index), g["mean"], g["size"], "Survival rate by fare quartile")
    fig.savefig(f"{OUTDIR}/04_fare.png", dpi=150)
    print("\n=== Fare quartile ===\n", g.round(3))

    # --- 5. 同乗家族 (SibSp / Parch / FamilySize) ---
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.8), layout="constrained")
    for ax, col in zip(axes, ["SibSp", "Parch", "FamilySize"]):
        v = df[col].clip(upper=5)
        labels = [str(i) if i < 5 else "5+" for i in sorted(v.unique())]
        g = df.groupby(v)["Survived"].agg(["mean", "size"])
        rate_bar(ax, labels, g["mean"], g["size"], f"Survival rate by {col}")
        ax.tick_params(axis="x", labelsize=9)
        print(f"\n=== {col} ===\n", g.round(3))
    fig.savefig(f"{OUTDIR}/05_family.png", dpi=150)

    # --- 6. Embarked / Cabin有無 ---
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), layout="constrained")
    g = df.groupby("Embarked")["Survived"].agg(["mean", "size"]).loc[["C", "Q", "S"]]
    rate_bar(
        axes[0], ["C (Cherbourg)", "Q (Queenstown)", "S (Southampton)"],
        g["mean"], g["size"], "Survival rate by Embarked",
    )
    axes[0].tick_params(axis="x", labelsize=9)
    print("\n=== Embarked ===\n", g.round(3))
    has_cabin = df["Cabin"].notna().map({True: "Has Cabin", False: "No Cabin"})
    g = df.groupby(has_cabin)["Survived"].agg(["mean", "size"]).loc[["Has Cabin", "No Cabin"]]
    rate_bar(axes[1], list(g.index), g["mean"], g["size"], "Survival rate by Cabin recorded")
    print("\n=== Cabin ===\n", g.round(3))
    fig.savefig(f"{OUTDIR}/06_embarked_cabin.png", dpi=150)

    # --- 単純ルールの当てはまり(train 上の一致率) ---
    print("\n=== 単純ルールの train 一致率 ===")
    rules = {
        "all died": np.zeros(len(df)),
        "female survived": (df.Sex == "female").astype(int),
        "female OR (child<10 & class<3)": (
            (df.Sex == "female") | ((df.Age < 10) & (df.Pclass < 3))
        ).astype(int),
        "female except class3-S": (
            (df.Sex == "female")
            & ~((df.Pclass == 3) & (df.Embarked == "S"))
        ).astype(int),
    }
    for name, pred in rules.items():
        print(f"  {name}: {(pred == df.Survived).mean():.4f}")


if __name__ == "__main__":
    main()
