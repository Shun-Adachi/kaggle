# CV(train)と LB(test)のデータ性質の差を測る。
# ① 属性分布の比較(カテゴリ構成・数値分布)+ KS 検定 / カイ二乗検定
# ② adversarial validation: 「この行は train か test か」を当てる分類器を CV で学習。
#    AUC ≈ 0.5 なら属性分布はほぼ同一(見分けられない)、高いなら分布シフトあり。
# ③ WCG 証拠(同グループの女性・子供の生死情報が届くか)のカバレッジ比較 — 既知の最大の差
# 出力: notebooks/figures/train_test_shift/<日付>/ に図4枚 + summary.json + feature_stats.csv
# 実行(リポジトリルートから): .venv/bin/python competitions/titanic/notebooks/train_test_shift.py
import json
import os
import sys
from datetime import date

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from scipy import stats
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score

sys.path.insert(0, "competitions/titanic/src")
from train import is_woman_child, load_train_test  # noqa: E402

OUTBASE = "competitions/titanic/notebooks/figures/train_test_shift"

# dataviz 参照パレット(error_analysis.py と共通)
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
C_TRAIN = "#2a78d6"  # blue: train
C_TEST = "#eb6834"   # orange: test

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
NUM_COLS = ["Age", "Fare", "FarePerPerson", "FamilySize", "TicketGroupSize"]
CAT_COLS = ["Pclass", "Sex", "Embarked", "Title", "Deck", "CabinFlag"]


def style_ax(ax, title):
    ax.set_title(title, color=INK, fontsize=11, loc="left", pad=8)
    ax.tick_params(length=0)
    ax.grid(axis="x", visible=False)


def grouped_share_bar(ax, tr_share, te_share, title, rotate=0):
    """train/test のカテゴリ構成比を並べる棒グラフ。"""
    cats = tr_share.index
    x = np.arange(len(cats))
    ax.bar(x - 0.18, tr_share * 100, width=0.32, color=C_TRAIN, label="train", zorder=2)
    ax.bar(x + 0.18, te_share * 100, width=0.32, color=C_TEST, label="test", zorder=2)
    ax.set_xticks(x, cats, rotation=rotate, ha="right" if rotate else "center")
    ax.set_ylabel("share (%)")
    style_ax(ax, title)


def fig_categorical(train_df, test_df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    seg_tr = (train_df["Sex"] + " C" + train_df["Pclass"].astype(str)).value_counts(normalize=True).reindex(SEG_ORDER)
    seg_te = (test_df["Sex"] + " C" + test_df["Pclass"].astype(str)).value_counts(normalize=True).reindex(SEG_ORDER)
    grouped_share_bar(axes[0, 0], seg_tr, seg_te, "Sex x Pclass composition", rotate=30)
    for ax, col in zip([axes[0, 1], axes[1, 0], axes[1, 1]], ["Title", "Embarked", "Deck"]):
        tr = train_df[col].fillna("NA").value_counts(normalize=True)
        te = test_df[col].fillna("NA").value_counts(normalize=True)
        cats = tr.index.union(te.index)
        # 出現頻度順(train 基準)で揃える
        cats = sorted(cats, key=lambda c: -tr.get(c, 0))
        grouped_share_bar(ax, tr.reindex(cats).fillna(0), te.reindex(cats).fillna(0),
                          f"{col} composition")
    axes[0, 0].legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.suptitle("Categorical composition: train (891) vs test (418)",
                 color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{outdir}/01_categorical.png", dpi=150)
    plt.close(fig)


def fig_numeric(train_df, test_df, outdir):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    specs = [
        ("Age", np.arange(0, 85, 5), None),
        ("FarePerPerson", np.arange(0, 60, 2.5), "fare per person (capped at 60)"),
        ("FamilySize", np.arange(0.5, 12, 1), None),
        ("TicketGroupSize", np.arange(0.5, 12, 1), None),
    ]
    for ax, (col, bins, note) in zip(axes.ravel(), specs):
        tr = train_df[col].clip(upper=bins[-1]).dropna()
        te = test_df[col].clip(upper=bins[-1]).dropna()
        ax.hist(tr, bins=bins, density=True, histtype="stepfilled",
                color=C_TRAIN, alpha=0.45, label="train", zorder=2)
        ax.hist(te, bins=bins, density=True, histtype="step",
                color=C_TEST, linewidth=2, label="test", zorder=3)
        ks = stats.ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        style_ax(ax, f"{note or col}  (KS p={ks.pvalue:.2f})")
        ax.set_ylabel("density")
    axes[0, 0].legend(frameon=False, labelcolor=INK2, fontsize=9)
    fig.suptitle("Numeric distributions: train vs test (KS p>0.05 = no detectable shift)",
                 color=INK, fontsize=13, x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{outdir}/02_numeric.png", dpi=150)
    plt.close(fig)


def adversarial_validation(train_df, test_df, outdir):
    """train/test を見分ける分類器。AUC ≈ 0.5 なら属性分布は同一とみなせる。"""
    cols_num = ["Age", "Fare", "FarePerPerson", "SibSp", "Parch",
                "FamilySize", "TicketGroupSize", "CabinFlag"]
    cols_cat = ["Pclass", "Sex", "Embarked", "Title", "Deck"]
    both = pd.concat([train_df.assign(is_test=0), test_df.assign(is_test=1)],
                     ignore_index=True)
    X = pd.get_dummies(both[cols_num + cols_cat], columns=cols_cat)
    y = both["is_test"].to_numpy()
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.05, num_leaves=31,
                         random_state=42, verbose=-1)
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    proba = cross_val_predict(clf, X, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, proba)

    clf.fit(X, y)
    imp = pd.Series(clf.feature_importances_, index=X.columns).sort_values()[-10:]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(np.arange(len(imp)), imp, height=0.62, color=C_TRAIN, zorder=2)
    ax.set_yticks(np.arange(len(imp)), imp.index)
    ax.set_xlabel("LightGBM split importance")
    verdict = "train/test indistinguishable" if auc < 0.55 else "shift detected"
    style_ax(ax, f"Adversarial validation  —  CV AUC = {auc:.3f} ({verdict})")
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    fig.tight_layout()
    fig.savefig(f"{outdir}/03_adversarial.png", dpi=150)
    plt.close(fig)
    return auc


def wcg_coverage(train_df, test_df, outdir):
    """WCG 証拠が届く人の割合: train(LOO: 自分以外に同グループの WC が train にいるか)
    vs test(同グループの WC が train にいるか)。"""
    key = "ExtGroupId"
    wc_train = train_df[is_woman_child(train_df)]
    wc_counts = wc_train.groupby(key).size()

    def cov_share(df, is_train):
        cnt = df[key].map(wc_counts).fillna(0)
        own = is_woman_child(df) if is_train else pd.Series(False, index=df.index)
        return (cnt - own.astype(int)) > 0

    rows = []
    for name, df, flag in [("train", train_df, True), ("test", test_df, False)]:
        c = cov_share(df, flag)
        seg = df["Sex"] + " C" + df["Pclass"].astype(str)
        rows.append(c.groupby(seg).mean().reindex(SEG_ORDER).rename(name))
    cov = pd.concat(rows, axis=1)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(SEG_ORDER))
    ax.bar(x - 0.18, cov["train"] * 100, width=0.32, color=C_TRAIN, label="train (LOO)", zorder=2)
    ax.bar(x + 0.18, cov["test"] * 100, width=0.32, color=C_TEST, label="test", zorder=2)
    for xi, (a, b) in enumerate(zip(cov["train"], cov["test"])):
        ax.text(xi - 0.18, a * 100 + 1, f"{a:.0%}", ha="center", color=INK, fontsize=9)
        ax.text(xi + 0.18, b * 100 + 1, f"{b:.0%}", ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, SEG_ORDER)
    ax.set_ylabel("share with WCG evidence (%)")
    ax.legend(frameon=False, labelcolor=INK2, fontsize=9)
    tr_all = (cov_share(train_df, True)).mean()
    te_all = (cov_share(test_df, False)).mean()
    gap_seg = (cov["train"] - cov["test"]).idxmax()
    style_ax(ax, f"WCG evidence coverage  —  train {tr_all:.0%} vs test {te_all:.0%} "
                 f"(largest gap: {gap_seg} {cov.loc[gap_seg, 'train']:.0%} -> {cov.loc[gap_seg, 'test']:.0%})")
    fig.tight_layout()
    fig.savefig(f"{outdir}/04_wcg_coverage.png", dpi=150)
    plt.close(fig)
    return {"train": round(float(tr_all), 3), "test": round(float(te_all), 3),
            "by_segment": cov.round(3).to_dict()}


def feature_stats(train_df, test_df, outdir):
    """数値: KS 検定 / カテゴリ: カイ二乗検定 の一覧表。"""
    rows = []
    for col in NUM_COLS:
        ks = stats.ks_2samp(train_df[col].dropna(), test_df[col].dropna())
        rows.append({"feature": col, "type": "numeric", "test": "KS",
                     "stat": round(ks.statistic, 3), "pvalue": round(ks.pvalue, 3),
                     "train_median": float(train_df[col].median()),
                     "test_median": float(test_df[col].median()),
                     "train_missing": round(train_df[col].isna().mean(), 3),
                     "test_missing": round(test_df[col].isna().mean(), 3)})
    for col in CAT_COLS:
        tab = pd.crosstab(
            pd.concat([train_df[col], test_df[col]]).fillna("NA"),
            np.r_[np.zeros(len(train_df)), np.ones(len(test_df))])
        chi = stats.chi2_contingency(tab)
        rows.append({"feature": col, "type": "categorical", "test": "chi2",
                     "stat": round(chi.statistic, 2), "pvalue": round(chi.pvalue, 3)})
    df = pd.DataFrame(rows)
    df.to_csv(f"{outdir}/feature_stats.csv", index=False)
    return df


def main() -> None:
    train_df, test_df = load_train_test()
    outdir = f"{OUTBASE}/{date.today():%Y%m%d}"
    os.makedirs(outdir, exist_ok=True)

    fig_categorical(train_df, test_df, outdir)
    fig_numeric(train_df, test_df, outdir)
    auc = adversarial_validation(train_df, test_df, outdir)
    cov = wcg_coverage(train_df, test_df, outdir)
    st = feature_stats(train_df, test_df, outdir)

    with open(f"{outdir}/summary.json", "w") as f:
        json.dump({"adversarial_auc": round(float(auc), 3), "wcg_coverage": cov},
                  f, indent=2, ensure_ascii=False)

    print(f"adversarial validation AUC: {auc:.3f}(0.5 = 見分け不能)")
    print(f"WCG 証拠カバレッジ: train {cov['train']:.1%} vs test {cov['test']:.1%}")
    print("\n=== 特徴量ごとの検定(p<0.05 なら分布差あり)===")
    print(st.to_string(index=False))
    print(f"\n図とサマリを保存: {outdir}/")


if __name__ == "__main__":
    main()
