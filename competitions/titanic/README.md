# Titanic - Machine Learning from Disaster

https://www.kaggle.com/competitions/titanic

乗客の属性データから生存(Survived: 0/1)を予測する二値分類。評価指標は Accuracy。

## 結果

CV 設定(コンペ内で統一): StratifiedKFold 5-fold, shuffle=True, seed=42, 指標 Accuracy

実験の採否判定は repeated CV(`src/repeated_cv.py`、分割 seed 0〜9 の平均)で行う。
単一 seed の CV は分割運で ±0.01〜0.02 ぶれるため、2σ(≒0.012)未満の差は同格とみなす。
セグメント別(Sex×Pclass)の効果測定は `src/segment_cv.py --configs <モデル/特徴セット> ...` を使う。

| 日付 | アプローチ | CV | LB |
| --- | --- | --- | --- |
| 2026-07-29 | ロジスティック回帰(基本特徴+Title+FamilySize) | 0.8283 | 0.76794 |
| 2026-07-29 | LightGBM(+Title別Age補完+Ticketグループ特徴) | 0.8339 | - |
| 2026-07-29 | LightGBM tuned + グループ運命連動特徴(lgbm_t/wcg) | 0.8541 | - |
| 2026-07-29 | 同上、集計を女性・子供に限定した本家WCG方式(lgbm_t/wcg2) | 0.8608 | 0.76794 |
| 2026-07-31 | 同上+姓ベース拡張グループ+cabin(lgbm_t/wcg2sc) | 0.843 (repeated) | **0.77751** |

これまでの実験の経緯と教訓は [docs/experiment-history.md](docs/experiment-history.md) にまとめている。
| 2026-07-29 | LightGBM(同上) | 0.8227 | - |
| 2026-07-29 | RandomForest(同上) | 0.8137 | - |

学習・評価は `src/train.py`(使い方はファイル冒頭の docstring 参照)。

## データ

```bash
kaggle competitions download -c titanic -p data
unzip -o data/titanic.zip -d data
```

- `train.csv` … 891 件(ラベルあり)
- `test.csv` … 418 件(予測対象)
- `gender_submission.csv` … 提出フォーマットのサンプル
