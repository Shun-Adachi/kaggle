# Sex・Pclass と他要素の相関を把握するための EDA
# 出力: notebooks/figures/rulebase/07〜09 の PNG と相関行列の標準出力
# 実行: リポジトリルートから .venv/bin/python competitions/titanic/notebooks/correlation_eda.py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

# rcParams・配色・ヘルパーは rulebase_eda と共通(import で rcParams が適用される)
from rulebase_eda import C_DIED, C_SURVIVED, DATA, INK, INK2, MUTED, OUTDIR, rate_bar, style_ax

SEED = 42
np.random.seed(SEED)

C_AQUA = "#1baf7a"  # categorical slot3
C_RED = "#e34948"  # diverging の暖色極
C_GRAY_MID = "#f0efec"  # diverging の中立midpoint (light)

# 相関ヒートマップ用 diverging カラーマップ: blue ↔ gray ↔ red
CMAP = LinearSegmentedColormap.from_list("div", [C_SURVIVED, C_GRAY_MID, C_RED])


def main():
    df = pd.read_csv(DATA)
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1

    # --- 7. Spearman 相関ヒートマップ ---
    num = pd.DataFrame(
        {
            "Survived": df["Survived"],
            "Female": (df["Sex"] == "female").astype(int),
            "Pclass": df["Pclass"],
            "Age": df["Age"],
            "Fare": df["Fare"],
            "SibSp": df["SibSp"],
            "Parch": df["Parch"],
            "FamilySize": df["FamilySize"],
            "HasCabin": df["Cabin"].notna().astype(int),
            "Embarked_C": (df["Embarked"] == "C").astype(int),
        }
    )
    corr = num.corr(method="spearman")
    print("=== Spearman 相関行列 ===")
    print(corr.round(2))

    fig, ax = plt.subplots(figsize=(8.6, 7.2), layout="constrained")
    n = len(corr)
    ax.imshow(corr.values, cmap=CMAP, vmin=-1, vmax=1)
    ax.set_xticks(range(n), corr.columns, rotation=45, ha="right", color=INK2)
    ax.set_yticks(range(n), corr.columns, color=INK2)
    for i in range(n):
        for j in range(n):
            v = corr.values[i, j]
            ax.text(
                j, i, f"{v:.2f}", ha="center", va="center", fontsize=9,
                color="#ffffff" if abs(v) > 0.55 else INK,
            )
    ax.grid(False)
    ax.tick_params(length=0)
    ax.set_title(
        "Spearman correlation (Female=1, HasCabin=1, Embarked_C=1)",
        color=INK, fontsize=12, loc="left", pad=10,
    )
    fig.savefig(f"{OUTDIR}/07_correlation_heatmap.png", dpi=150)

    # --- 8. Pclass と他要素 ---
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")
    classes = [1, 2, 3]
    labels = [f"Class {c}" for c in classes]
    x = np.arange(3)

    ax = axes[0]  # Fare: 中央値 + IQR
    g = df.groupby("Pclass")["Fare"]
    med, q1, q3 = g.median(), g.quantile(0.25), g.quantile(0.75)
    ax.bar(x, med, width=0.6, color=C_SURVIVED, zorder=2)
    ax.errorbar(
        x, med, yerr=[med - q1, q3 - med], fmt="none",
        ecolor=INK2, elinewidth=1.5, capsize=4, zorder=3,
    )
    for xi, m in zip(x, med):
        ax.text(xi + 0.33, m + 2, f"{m:.0f}", ha="left", color=INK, fontsize=10)
    ax.set_xticks(x, labels)
    style_ax(ax, "Fare by Pclass (median, IQR)")
    print("\n=== Fare by Pclass(中央値 [Q1-Q3])===")
    for c in classes:
        print(f"  Class {c}: {med[c]:.1f} [{q1[c]:.1f}-{q3[c]:.1f}]")

    ax = axes[1]  # Age: 平均
    g = df.dropna(subset=["Age"]).groupby("Pclass")["Age"]
    ax.bar(x, g.mean(), width=0.6, color=C_SURVIVED, zorder=2)
    for xi, m in zip(x, g.mean()):
        ax.text(xi, m + 1, f"{m:.0f}", ha="center", color=INK, fontsize=10)
    ax.set_xticks(x, [f"{l}\nn={c}" for l, c in zip(labels, g.size())])
    style_ax(ax, "Mean age by Pclass")
    print("\n=== Mean Age by Pclass ===\n", g.mean().round(1).to_dict())

    ax = axes[2]  # Cabin 記録率
    g = df.groupby("Pclass")["Cabin"].agg(lambda s: s.notna().mean())
    ns = df.groupby("Pclass").size()
    rate_bar(ax, labels, g, ns, "Cabin recorded rate by Pclass")
    print("\n=== Cabin 記録率 by Pclass ===\n", g.round(3).to_dict())

    ax = axes[3]  # Embarked 構成比(グループ棒)
    comp = pd.crosstab(df["Pclass"], df["Embarked"], normalize="index")[["C", "Q", "S"]]
    w = 0.26
    for k, (port, color) in enumerate(zip(["C", "Q", "S"], [C_SURVIVED, C_DIED, C_AQUA])):
        ax.bar(x + (k - 1) * w, comp[port], w - 0.02, color=color, label=port, zorder=2)
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.legend(frameon=False, labelcolor=INK2, title="Embarked", title_fontsize=9)
    style_ax(ax, "Embarked share by Pclass")
    print("\n=== Embarked 構成比 by Pclass ===\n", comp.round(3))
    fig.savefig(f"{OUTDIR}/08_pclass_vs_others.png", dpi=150)

    # 構成比のグループ棒(series: ラベル→(色, 値列))
    def comp_bars(ax, xlabels, series, title, ns=None, legend_title=None):
        x = np.arange(len(xlabels))
        k = len(series)
        w = 0.8 / k
        for i, (name, (color, vals)) in enumerate(series.items()):
            ax.bar(x + (i - (k - 1) / 2) * w, vals, w - 0.02, color=color, label=name, zorder=2)
            for xi, v in zip(x, vals):
                ax.text(
                    xi + (i - (k - 1) / 2) * w, v + 0.015, f"{v:.0%}",
                    ha="center", color=INK, fontsize=8,
                )
        ticks = [f"{l}\nn={n}" for l, n in zip(xlabels, ns)] if ns is not None else xlabels
        ax.set_xticks(x, ticks)
        ax.set_ylim(0, 1.0)
        ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
        ax.legend(frameon=False, labelcolor=INK2, title=legend_title, title_fontsize=9)
        style_ax(ax, title)

    # --- 9. Sex と他要素(男女両方の構成比) ---
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), layout="constrained")

    sex_cls = pd.crosstab(df["Pclass"], df["Sex"], normalize="index")
    comp_bars(
        axes[0], labels,
        {"female": (C_SURVIVED, sex_cls["female"]), "male": (C_DIED, sex_cls["male"])},
        "Sex share by Pclass", ns=df.groupby("Pclass").size(),
    )
    print("\n=== 男女構成比 by Pclass ===\n", sex_cls.round(3))

    ax = axes[1]  # 単身率 by Sex
    alone = df.assign(Alone=(df.FamilySize == 1)).groupby("Sex")["Alone"].agg(["mean", "size"])
    rate_bar(
        ax, ["female", "male"],
        alone.loc[["female", "male"], "mean"], alone.loc[["female", "male"], "size"],
        "Traveling alone rate by Sex",
    )
    print("\n=== 単身率 by Sex ===\n", alone.round(3))

    ax = axes[2]  # 年齢分布 by Sex
    a = df.dropna(subset=["Age"])
    ax.hist(
        [a.loc[a.Sex == "female", "Age"], a.loc[a.Sex == "male", "Age"]],
        bins=np.arange(0, 85, 5), color=[C_SURVIVED, C_DIED],
        label=["female", "male"], zorder=2, rwidth=0.92,
    )
    ax.legend(frameon=False, labelcolor=INK2)
    ax.set_xlabel("Age")
    style_ax(ax, "Age distribution by Sex")

    sex_emb = pd.crosstab(df["Embarked"], df["Sex"], normalize="index").loc[["C", "Q", "S"]]
    comp_bars(
        axes[3], ["C", "Q", "S"],
        {"female": (C_SURVIVED, sex_emb["female"]), "male": (C_DIED, sex_emb["male"])},
        "Sex share by Embarked", ns=df.groupby("Embarked").size()[["C", "Q", "S"]],
    )
    print("\n=== 男女構成比 by Embarked ===\n", sex_emb.round(3))
    fig.savefig(f"{OUTDIR}/09_sex_vs_others.png", dpi=150)

    # --- 10. 家族構成 × 性別・等級 ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), layout="constrained")
    fs = df["FamilySize"].clip(upper=5)
    fs_labels = ["1", "2", "3", "4", "5+"]

    fam_sex = pd.crosstab(fs, df["Sex"], normalize="columns")
    comp_bars(
        axes[0], fs_labels,
        {"female": (C_SURVIVED, fam_sex["female"]), "male": (C_DIED, fam_sex["male"])},
        "FamilySize distribution by Sex",
    )
    axes[0].set_xlabel("FamilySize")
    print("\n=== FamilySize 構成比(性別内)===\n", fam_sex.round(3))

    fam_cls = pd.crosstab(fs, df["Pclass"], normalize="columns")
    comp_bars(
        axes[1], fs_labels,
        {f"Class {c}": (col, fam_cls[c]) for c, col in zip(classes, [C_SURVIVED, C_DIED, C_AQUA])},
        "FamilySize distribution by Pclass",
    )
    axes[1].set_xlabel("FamilySize")
    print("\n=== FamilySize 構成比(クラス内)===\n", fam_cls.round(3))

    alone_sc = df.assign(Alone=(df.FamilySize == 1)).pivot_table(
        index="Pclass", columns="Sex", values="Alone", aggfunc="mean"
    )
    comp_bars(
        axes[2], labels,
        {"female": (C_SURVIVED, alone_sc["female"]), "male": (C_DIED, alone_sc["male"])},
        "Traveling alone rate by Sex × Pclass",
    )
    print("\n=== 単身率 by Sex × Pclass ===\n", alone_sc.round(3))
    fig.savefig(f"{OUTDIR}/10_family_vs_sex_pclass.png", dpi=150)


if __name__ == "__main__":
    main()
