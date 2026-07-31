"""repeated CV: CV 分割の seed を変えて主要構成を評価し、分割運・選択バイアスを除いた実力値を測る。

実行(リポジトリルートから):
    .venv/bin/python competitions/titanic/src/repeated_cv.py

- seed 0〜9 の10通り(パラメータ選択に使った seed 42 は意図的に除外)。
- 各 seed で OOF accuracy を1つ算出し、構成ごとに平均・標準偏差・最小/最大を出す。
- 参考列として seed 42(選択に使った分割)の値も併記し、上振れ幅を可視化する。
"""

import numpy as np

from train import fold_scores, load_train_test, run_cv

SEEDS = list(range(10))  # 選択に使った 42 は除外
CONFIGS = [
    ("logreg", "base"),
    ("logreg", "full"),
    ("lgbm", "full"),
    ("lgbm_t", "wcg"),
    ("lgbm_t", "wcg2"),
    ("lgbm_t", "wcg3"),
]


def oof_acc(model, fs, train_df, seed):
    oof = run_cv(model, fs, train_df, seed=seed)
    return ((oof >= 0.5).astype(int) == train_df["Survived"].to_numpy()).mean()


def main() -> None:
    train_df, _ = load_train_test()
    print(f"{'構成':24s} {'10seed平均':>10s} {'std':>7s} {'min':>7s} {'max':>7s} {'seed42':>7s}")
    for model, fs in CONFIGS:
        accs = np.array([oof_acc(model, fs, train_df, s) for s in SEEDS])
        ref42 = oof_acc(model, fs, train_df, 42)
        print(
            f"{model + '/' + fs:24s} {accs.mean():>10.4f} {accs.std():>7.4f}"
            f" {accs.min():>7.4f} {accs.max():>7.4f} {ref42:>7.4f}"
        )


if __name__ == "__main__":
    main()
