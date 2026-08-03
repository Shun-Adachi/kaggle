"""Titanic: 前処理 + モデル定義 + CV 評価 + submission 生成。

使い方(リポジトリルートから):
    .venv/bin/python competitions/titanic/src/train.py                          # 3モデルを CV で比較(特徴セット full)
    .venv/bin/python competitions/titanic/src/train.py --features wcg          # 特徴セットを変えて比較(ablation)
    .venv/bin/python competitions/titanic/src/train.py --model lgbm            # 1モデルを CV + 全データ学習 + submission 出力

特徴セット(積み上げ式):
    base   … 生7列 + Title + FamilySize、Age は全体中央値で補完(初回ベースライン)
    age    … base + Age を Title 別中央値で補完
    ticket … base + TicketGroupSize / FarePerPerson
    full   … age + ticket 全部入り
    cabin  … full + CabinFlag(客室記録の有無)+ Deck(客室のデッキ文字)
    bins   … full + FamilyBand / GroupBand(人数の非線形をカテゴリ化。線形モデル向け)
    wcg    … full + GroupSurvRate / GroupInfo(同一チケット同行者の生死。リーク対策済み)
    v2     … wcg + cabin
    v3     … v2 + bins
    wcg2   … full + WcGroupSurvRate / WcGroupInfo(集計を女性・子供に限定した本家 WCG 方式)
    wcg3   … full + 全員版とWC版の両方

グループ運命連動特徴(wcg)のリーク対策:
    同行者の生死は正解ラベル由来なので、CV では各フォールドの学習側ラベルだけから計算する。
    学習側の行自身は leave-one-out(自分を除いた同行者のみ)。情報が無い場合は
    学習側全体の生存率で埋め、GroupInfo=0 で「情報なし」を明示する。

CV 設定(コンペ内で統一): StratifiedKFold 5-fold, shuffle=True, seed=42, 指標 Accuracy
"""

import argparse
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 42
COMP_DIR = Path(__file__).resolve().parents[1]

BASE_NUM = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
TICKET_NUM = ["TicketGroupSize", "FarePerPerson"]
GROUP_NUM = ["GroupSurvRate", "GroupInfo"]
WC_NUM = ["WcGroupSurvRate", "WcGroupInfo"]
BASE_CAT = ["Pclass", "Sex", "Embarked", "Title"]

# 特徴セット名 → 列構成・オプション
FEATURE_SETS = {
    "base": {"num": BASE_NUM, "cat": BASE_CAT, "title_age": False, "group": False},
    "age": {"num": BASE_NUM, "cat": BASE_CAT, "title_age": True, "group": False},
    "ticket": {"num": BASE_NUM + TICKET_NUM, "cat": BASE_CAT, "title_age": False, "group": False},
    "full": {"num": BASE_NUM + TICKET_NUM, "cat": BASE_CAT, "title_age": True, "group": False},
    "cabin": {
        "num": BASE_NUM + TICKET_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck"], "title_age": True, "group": False,
    },
    "bins": {
        "num": BASE_NUM + TICKET_NUM,
        "cat": BASE_CAT + ["FamilyBand", "GroupBand"], "title_age": True, "group": False,
    },
    "wcg": {
        "num": BASE_NUM + TICKET_NUM + GROUP_NUM,
        "cat": BASE_CAT, "title_age": True, "group": "all",
    },
    "v2": {
        "num": BASE_NUM + TICKET_NUM + GROUP_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck"], "title_age": True, "group": "all",
    },
    "v3": {
        "num": BASE_NUM + TICKET_NUM + GROUP_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck", "FamilyBand", "GroupBand"], "title_age": True, "group": "all",
    },
    # 本家 Woman-Child Group 方式: 同行者のうち女性・子供(female または Master)だけを集計
    "wcg2": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM,
        "cat": BASE_CAT, "title_age": True, "group": "wc",
    },
    "wcg3": {  # 全員版と女性・子供版の両方
        "num": BASE_NUM + TICKET_NUM + GROUP_NUM + WC_NUM,
        "cat": BASE_CAT, "title_age": True, "group": "both",
    },
    # wcg2 の拡張: グループを「同一チケット ∪ 同姓家族(別チケット)」に広げて証拠の届く範囲を増やす
    "wcg2s": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM,
        "cat": BASE_CAT, "title_age": True, "group": "wc", "group_key": "ExtGroupId",
    },
    # wcg2 + cabin(male C1 局所で有効か検証用)
    "wcg2c": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck"], "title_age": True, "group": "wc",
    },
    # 両方
    "wcg2sc": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck"], "title_age": True, "group": "wc", "group_key": "ExtGroupId",
    },
    # male C1 深掘り(2026-08-03): 大人の男性に WCG 証拠を渡さない(rate=1 の誤読による FP 防止)
    "wcg2scm": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck"], "title_age": True, "group": "wc", "group_key": "ExtGroupId",
        "wc_mask_men": True,
    },
    # male C1 深掘り: Age 欠損フラグ(欠損者は低生存率なのに中央値補完で高生存帯に混入する問題)
    "wcg2sca": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag", "AgeMissing"],
        "cat": BASE_CAT + ["Deck"], "title_age": True, "group": "wc", "group_key": "ExtGroupId",
    },
    # 上2つの併用
    "wcg2scma": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag", "AgeMissing"],
        "cat": BASE_CAT + ["Deck"], "title_age": True, "group": "wc", "group_key": "ExtGroupId",
        "wc_mask_men": True,
    },
    # 証拠は消さず「女性・子供か」の条件分岐だけ明示する案(ペア比較 全体+2.2人・7勝2敗の傾向。
    # LB 検証で 2勝4敗 → 棄却)
    "wcg2scw": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag", "IsWC"],
        "cat": BASE_CAT + ["Deck"], "title_age": True, "group": "wc", "group_key": "ExtGroupId",
    },
    # Age 補完の精緻化(2026-08-03): Title×Pclass 中央値(MAE 9.41→8.93歳。
    # 欠損の Mr C1 が 30歳→40歳になり実際の低生存率と整合する帯に入る)
    "wcg2scp": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck"], "title_age": "pclass", "group": "wc", "group_key": "ExtGroupId",
    },
    # Age 補完の精緻化: LGBM 回帰補完(MAE 8.62歳)
    "wcg2scr": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck"], "title_age": "reg", "group": "wc", "group_key": "ExtGroupId",
    },
    # Age 補完の精緻化: 親同伴の Miss を女児として補完(TitleFine 別中央値)
    "wcg2scf": {
        "num": BASE_NUM + TICKET_NUM + WC_NUM + ["CabinFlag"],
        "cat": BASE_CAT + ["Deck"], "title_age": "fine", "group": "wc", "group_key": "ExtGroupId",
    },
}

TITLE_MAP = {"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"}
COMMON_TITLES = ["Mr", "Miss", "Mrs", "Master"]

# チューニング結果(src/tune_lgbm.py で特徴セットごとに探索した値をここに固定する。
# 未探索のセットで lgbm_t を指定した場合はデフォルトパラメータになる)
_WCG2_PARAMS = {"n_estimators": 100, "learning_rate": 0.03, "num_leaves": 15, "min_child_samples": 20}
LGBM_TUNED = {
    "wcg": {"n_estimators": 100, "learning_rate": 0.03, "num_leaves": 31, "min_child_samples": 20},   # CV 0.8541
    "wcg2": _WCG2_PARAMS,  # CV 0.8608(seed42)
    "wcg3": _WCG2_PARAMS,  # CV 0.8563(seed42)
    # 派生セットは wcg2 の値を流用(セットごとに再グリッドすると選択バイアスが積むため)
    "wcg2s": _WCG2_PARAMS,
    "wcg2c": _WCG2_PARAMS,
    "wcg2sc": _WCG2_PARAMS,
    "wcg2scm": _WCG2_PARAMS,
    "wcg2sca": _WCG2_PARAMS,
    "wcg2scma": _WCG2_PARAMS,
    "wcg2scw": _WCG2_PARAMS,
    "wcg2scp": _WCG2_PARAMS,
    "wcg2scr": _WCG2_PARAMS,
    "wcg2scf": _WCG2_PARAMS,
}


class TitleAgeImputer(BaseEstimator, TransformerMixin):
    """Age の欠損を Title(既定)または Title×Pclass 別中央値で埋める。

    中央値は fit したデータ(CV では学習フォールドのみ)から学ぶことでリークを防ぐ。
    セルが無い場合は Title 別 → 全体中央値へフォールバック。
    Title×Pclass は補完 MAE 9.41 → 8.93 歳(Mr は C1=40 / C2=31 / C3=26 歳と階級差が大きい)。
    """

    def __init__(self, by=("Title",)):
        self.by = by

    @staticmethod
    def _with_derived(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        # TitleFine: 親同伴の Miss は女児の可能性が高い(年齢既知では中央値9歳。
        # 一律 Miss=21歳で埋めると子供が大人扱いになる)。パイプラインには
        # 特徴セットの列しか渡らないため、Title と Parch からここで導出する
        if "TitleFine" not in X.columns:
            X["TitleFine"] = X["Title"].where(
                ~((X["Title"] == "Miss") & (X["Parch"] > 0)), "MissChild")
        return X

    def fit(self, X, y=None):
        X = self._with_derived(X)
        self.medians_ = X.groupby(list(self.by))["Age"].median()
        self.title_medians_ = X.groupby("Title")["Age"].median()
        self.fallback_ = X["Age"].median()
        return self

    def transform(self, X):
        X = self._with_derived(X)
        if len(self.by) == 1:
            fill = X[self.by[0]].map(self.medians_)
        else:
            key = pd.MultiIndex.from_frame(X[list(self.by)])
            fill = pd.Series(self.medians_.reindex(key).to_numpy(), index=X.index)
        fill = fill.fillna(X["Title"].map(self.title_medians_)).fillna(self.fallback_)
        X["Age"] = X["Age"].fillna(fill)
        return X.drop(columns=["TitleFine"])


class RegAgeImputer(BaseEstimator, TransformerMixin):
    """Age の欠損を他の属性からの LightGBM 回帰で補完する(補完 MAE 8.62 歳)。

    回帰モデルは fit したデータの Age 既知行のみから学習(リーク防止は Title 版と同じ理屈)。
    """

    COLS = ["Pclass", "SibSp", "Parch", "Fare", "FamilySize", "TicketGroupSize", "FarePerPerson"]
    CATS = {"Title": ["Mr", "Miss", "Mrs", "Master", "Rare"],
            "Sex": ["male", "female"], "Embarked": ["S", "C", "Q"]}

    def _enc(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.COLS].copy()
        for c, vals in self.CATS.items():
            for v in vals:
                out[f"{c}_{v}"] = (X[c] == v).astype(int)
        return out

    def fit(self, X, y=None):
        from lightgbm import LGBMRegressor
        known = X[X["Age"].notna()]
        self.model_ = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=15,
                                    random_state=SEED, verbose=-1)
        self.model_.fit(self._enc(known), known["Age"])
        return self

    def transform(self, X):
        X = X.copy()
        miss = X["Age"].isna()
        if miss.any():
            X.loc[miss, "Age"] = self.model_.predict(self._enc(X[miss]))
        return X


def add_features(df: pd.DataFrame, ticket_counts: pd.Series) -> pd.DataFrame:
    out = df.copy()
    title = out["Name"].str.extract(r" ([A-Za-z]+)\.", expand=False)
    title = title.replace(TITLE_MAP)
    out["Title"] = title.where(title.isin(COMMON_TITLES), "Rare")
    out["FamilySize"] = out["SibSp"] + out["Parch"] + 1
    # Age が記録されていないこと自体が情報(male C1 では欠損者の生存率 23.8% に対し、
    # Title別中央値補完で 30歳=最も生存率の高い帯に混入してしまう)
    out["AgeMissing"] = out["Age"].isna().astype(int)
    # WCG 証拠の効き方は女性・子供と大人の男性で別(rate=1 でも大人の男性の生存率 27.7%)。
    # その条件分岐を木に1分割で学ばせるための明示フラグ
    out["IsWC"] = ((out["Sex"] == "female") | (out["Title"] == "Master")).astype(int)
    # 同一チケット = 同行グループ。人数は train+test 合算で数える(ラベルは使わない)
    out["TicketGroupSize"] = out["Ticket"].map(ticket_counts)
    out["FarePerPerson"] = out["Fare"] / out["TicketGroupSize"]
    # Cabin: 77% 欠損だが「記録あり」自体が上級客室の印。デッキ文字も抽出(T は1名のみ→U 扱い)
    out["CabinFlag"] = out["Cabin"].notna().astype(int)
    out["Deck"] = out["Cabin"].str[0].replace({"T": "U"}).fillna("U")
    # 人数の非線形(2〜4人で山、5人以上で谷)をカテゴリ化。線形モデル向け
    out["FamilyBand"] = pd.cut(out["FamilySize"], [0, 1, 4, 20], labels=["solo", "small", "large"])
    out["GroupBand"] = pd.cut(out["TicketGroupSize"], [0, 1, 4, 20], labels=["solo", "small", "large"])
    return out


def is_woman_child(df: pd.DataFrame) -> pd.Series:
    return (df["Sex"] == "female") | (df["Title"] == "Master")


def _extended_group_ids(all_df: pd.DataFrame) -> np.ndarray:
    """同一チケット ∪ 同姓家族(姓+Pclass+Embarked が同じで家族欄あり)を連結したグループ ID。

    姓だけで繋ぐと無関係の同姓(3等の Andersson 単身者など)を誤結合するため、
    FamilySize > 1(家族連れと申告している人)同士に限って姓リンクを張る。
    """
    parent = list(range(len(all_df)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union_all(idxs) -> None:
        first = find(idxs[0])
        for i in idxs[1:]:
            parent[find(i)] = first

    for idxs in all_df.groupby("Ticket").indices.values():
        union_all(list(idxs))
    fam = all_df["FamilySize"].to_numpy() > 1
    surname_key = all_df.groupby(
        ["Surname", "Pclass", all_df["Embarked"].fillna("?")]
    ).indices
    for idxs in surname_key.values():
        linked = [i for i in idxs if fam[i]]
        if len(linked) >= 2:
            union_all(linked)
    return np.array([find(i) for i in range(len(all_df))])


def add_group_feature(train_part: pd.DataFrame, target: pd.DataFrame,
                      wc_only: bool = False, key: str = "Ticket",
                      mask_non_wc: bool = False) -> pd.DataFrame:
    """同行グループ(key 列が同じ人)の生存率と情報有無を付与する。

    統計は train_part のラベルのみから計算。target が集計対象内の行なら自分を除く(LOO)。
    wc_only=True では集計対象を女性・子供(female または Master)に限定し、
    WcGroupSurvRate / WcGroupInfo 列に書く(本家 Woman-Child Group 方式)。
    key は "Ticket"(同一チケット)または "ExtGroupId"(チケット∪同姓家族)。
    """
    pool = train_part[is_woman_child(train_part)] if wc_only else train_part
    prefix = "Wc" if wc_only else ""
    stats = pool.groupby(key)["Survived"].agg(["sum", "count"])
    out = target.copy()
    joined = out.join(stats, on=key)
    # 自分自身の除外(LOO)判定は PassengerId で行う。index だと train(0〜890)と
    # test(0〜417)の番号が衝突し、test 行が誤って「自分がプールにいる」扱いになる
    # (このバグで test の24人が証拠を失い37人の rate がずれていた。CV は fold 間で
    # index が重複しないため無影響)
    in_pool = out["PassengerId"].isin(pool["PassengerId"]).to_numpy().astype(float)
    own = out["Survived"].to_numpy(dtype=float) if "Survived" in out.columns else np.zeros(len(out))
    cnt = joined["count"].fillna(0).to_numpy() - in_pool
    ssum = joined["sum"].fillna(0).to_numpy() - np.nan_to_num(own) * in_pool
    rate = np.divide(ssum, cnt, out=np.full(len(out), np.nan), where=cnt > 0)
    out[f"{prefix}GroupSurvRate"] = np.where(np.isnan(rate), pool["Survived"].mean(), rate)
    out[f"{prefix}GroupInfo"] = (cnt > 0).astype(int)
    if mask_non_wc:
        # 大人の男性には証拠を渡さない(女性・子供の生死と大人の男性の生死は連動が弱く、
        # rate=1 のグループでも大人の男性の生存率は 27.7%。モデルが rate=1 を
        # 「生存の証拠」と誤読して FP を作るのを防ぐ)
        men = ~is_woman_child(out)
        out.loc[men, f"{prefix}GroupSurvRate"] = pool["Survived"].mean()
        out.loc[men, f"{prefix}GroupInfo"] = 0
    return out


def apply_group(fs: dict, train_part: pd.DataFrame, target: pd.DataFrame) -> pd.DataFrame:
    mode = fs["group"]
    key = fs.get("group_key", "Ticket")
    if mode in ("all", "both"):
        target = add_group_feature(train_part, target, wc_only=False, key=key)
    if mode in ("wc", "both"):
        target = add_group_feature(train_part, target, wc_only=True, key=key,
                                   mask_non_wc=fs.get("wc_mask_men", False))
    return target


def build_model(name: str, feature_set: str, params: dict | None = None) -> Pipeline:
    fs = FEATURE_SETS[feature_set]
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                fs["num"],
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                fs["cat"],
            ),
        ]
    )
    estimators = {
        "logreg": lambda: LogisticRegression(max_iter=1000, random_state=SEED),
        "rf": lambda: RandomForestClassifier(n_estimators=400, random_state=SEED),
        "lgbm": lambda: LGBMClassifier(random_state=SEED, verbose=-1, **(params or {})),
        "lgbm_t": lambda: LGBMClassifier(
            random_state=SEED, verbose=-1, **(params or LGBM_TUNED.get(feature_set, {}))
        ),
    }
    steps = [("prep", preprocessor), ("clf", estimators[name]())]
    if fs["title_age"] == "pclass":
        steps.insert(0, ("age", TitleAgeImputer(by=("Title", "Pclass"))))
    elif fs["title_age"] == "fine":
        steps.insert(0, ("age", TitleAgeImputer(by=("TitleFine",))))
    elif fs["title_age"] == "reg":
        steps.insert(0, ("age", RegAgeImputer()))
    elif fs["title_age"]:
        steps.insert(0, ("age", TitleAgeImputer()))
    return Pipeline(steps)


def load_train_test() -> tuple[pd.DataFrame, pd.DataFrame]:
    train_raw = pd.read_csv(COMP_DIR / "data" / "train.csv")
    test_raw = pd.read_csv(COMP_DIR / "data" / "test.csv")
    ticket_counts = pd.concat([train_raw["Ticket"], test_raw["Ticket"]]).value_counts()
    train_df = add_features(train_raw, ticket_counts)
    test_df = add_features(test_raw, ticket_counts)
    # 拡張グループ ID は train+test を通しで計算しないと一貫しない
    all_df = pd.concat([train_df, test_df], ignore_index=True)
    all_df["Surname"] = all_df["Name"].str.split(",").str[0]
    gids = _extended_group_ids(all_df)
    train_df["ExtGroupId"] = gids[: len(train_df)]
    test_df["ExtGroupId"] = gids[len(train_df):]
    return train_df, test_df


def make_cv(seed: int = SEED) -> StratifiedKFold:
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)


def run_cv(model_name: str, fs_name: str, train_df: pd.DataFrame,
           params: dict | None = None, seed: int = SEED) -> np.ndarray:
    """OOF 予測確率を返す(グループ特徴は各フォールドの学習側ラベルのみで計算)。

    seed は CV 分割の乱数。既定は SEED=42(実験の標準設定)。
    repeated CV で分割運の影響を測るときだけ変える。
    """
    if model_name == "ens":  # logreg と lgbm_t の確率平均
        return (run_cv("logreg", fs_name, train_df, seed=seed)
                + run_cv("lgbm_t", fs_name, train_df, seed=seed)) / 2
    if model_name == "avg3":  # logreg・rf・lgbm_t の確率平均(repeated CV ペア比較で10seed全勝)
        return (run_cv("logreg", fs_name, train_df, seed=seed)
                + run_cv("rf", fs_name, train_df, seed=seed)
                + run_cv("lgbm_t", fs_name, train_df, seed=seed)) / 3
    fs = FEATURE_SETS[fs_name]
    cols = fs["num"] + fs["cat"]
    y = train_df["Survived"].to_numpy()
    oof = np.zeros(len(train_df))
    for tr_idx, va_idx in make_cv(seed).split(train_df, y):
        tr, va = train_df.iloc[tr_idx], train_df.iloc[va_idx]
        if fs["group"]:
            tr, va = apply_group(fs, tr, tr), apply_group(fs, tr, va)
        model = build_model(model_name, fs_name, params)
        model.fit(tr[cols], y[tr_idx])
        oof[va_idx] = model.predict_proba(va[cols])[:, 1]
    return oof


def fold_scores(oof: np.ndarray, train_df: pd.DataFrame, seed: int = SEED) -> np.ndarray:
    y = train_df["Survived"].to_numpy()
    pred = (oof >= 0.5).astype(int)
    return np.array([(pred[va] == y[va]).mean() for _, va in make_cv(seed).split(train_df, y)])


def predict_test(model_name: str, fs_name: str, train_df: pd.DataFrame,
                 test_df: pd.DataFrame, params: dict | None = None) -> np.ndarray:
    """全 train で学習して test の生存確率を返す。"""
    if model_name == "ens":
        return (predict_test("logreg", fs_name, train_df, test_df)
                + predict_test("lgbm_t", fs_name, train_df, test_df)) / 2
    if model_name == "avg3":
        return (predict_test("logreg", fs_name, train_df, test_df)
                + predict_test("rf", fs_name, train_df, test_df)
                + predict_test("lgbm_t", fs_name, train_df, test_df)) / 3
    fs = FEATURE_SETS[fs_name]
    cols = fs["num"] + fs["cat"]
    tr, te = train_df, test_df
    if fs["group"]:
        tr, te = apply_group(fs, train_df, train_df), apply_group(fs, train_df, test_df)
    model = build_model(model_name, fs_name, params)
    model.fit(tr[cols], train_df["Survived"].to_numpy())
    return model.predict_proba(te[cols])[:, 1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        choices=["all", "logreg", "rf", "lgbm", "lgbm_t", "ens", "avg3"],
        default="all",
        help="all は CV 比較のみ。個別指定で submission も出力",
    )
    parser.add_argument(
        "--features",
        choices=list(FEATURE_SETS),
        default="full",
        help="使う特徴セット(ablation 用)",
    )
    args = parser.parse_args()

    train_df, test_df = load_train_test()
    names = ["logreg", "rf", "lgbm"] if args.model == "all" else [args.model]
    for name in names:
        oof = run_cv(name, args.features, train_df)
        scores = fold_scores(oof, train_df)
        folds = " ".join(f"{s:.4f}" for s in scores)
        print(
            f"[{args.features:6s}] {name:8s} CV accuracy: "
            f"{scores.mean():.4f} +/- {scores.std():.4f}  (folds: {folds})"
        )

        if args.model != "all":
            proba = predict_test(name, args.features, train_df, test_df)
            sub = pd.DataFrame(
                {"PassengerId": test_df["PassengerId"], "Survived": (proba >= 0.5).astype(int)}
            )
            out_path = (
                COMP_DIR
                / "submissions"
                / f"{date.today():%Y%m%d}_{name}_{args.features}_cv{scores.mean():.4f}.csv"
            )
            sub.to_csv(out_path, index=False)
            print(f"submission written: {out_path}")


if __name__ == "__main__":
    main()
