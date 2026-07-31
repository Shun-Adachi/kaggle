"""LightGBM の小規模グリッドサーチ(CV は train.py と同一設定)。

実行(リポジトリルートから):
    .venv/bin/python competitions/titanic/src/tune_lgbm.py [--features wcg]

小データ(891件)なので「浅め・強正則化」方向を中心に探索する。
ベストのパラメータは train.py の LGBM_TUNED に手で固定する(実験の再現性のため)。
"""

import argparse
import itertools

from train import FEATURE_SETS, fold_scores, load_train_test, run_cv

GRID = {
    "n_estimators": [100, 200, 400],
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [7, 15, 31],
    "min_child_samples": [10, 20],
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", default="wcg", choices=list(FEATURE_SETS))
    args = parser.parse_args()

    train_df, _ = load_train_test()
    results = []
    keys = list(GRID)
    for values in itertools.product(*GRID.values()):
        params = dict(zip(keys, values))
        scores = fold_scores(run_cv("lgbm", args.features, train_df, params), train_df)
        results.append((scores.mean(), scores.std(), params))
    results.sort(key=lambda r: -r[0])
    print(f"features={args.features}  上位10件(全{len(results)}通り):")
    for mean, std, params in results[:10]:
        print(f"  {mean:.4f} +/- {std:.4f}  {params}")


if __name__ == "__main__":
    main()
