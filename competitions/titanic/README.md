# Titanic - Machine Learning from Disaster

https://www.kaggle.com/competitions/titanic

乗客の属性データから生存(Survived: 0/1)を予測する二値分類。評価指標は Accuracy。

## 結果

CV 設定(コンペ内で統一): StratifiedKFold 5-fold, shuffle=True, seed=42, 指標 Accuracy

実験の採否判定は repeated CV(`src/repeated_cv.py`、分割 seed 0〜9 の平均)で行う。
単一 seed の CV は分割運で ±0.01〜0.02 ぶれるため、2σ(≒0.012)未満の差は同格とみなす。
ただし2構成の比較は**同一 seed 同士のペア差**で判定するとより高分解能
(分割運が共通化されるため。例: avg3 vs lgbm_t は平均差 +0.005 でも 10 seed 全勝 = 本物)。
ペア差で本物でも LB に転写されるとは限らない(avg3 は LB で 2勝12敗の -0.024)。
**提出前に「どの乗客の予測が変わるか」を diff し、変わる層が運の層(male C1 等)なら
その CV 改善は持ち出せない**とみなすこと。
セグメント別(Sex×Pclass)の効果測定は `src/segment_cv.py --configs <モデル/特徴セット> ...` を使う。

| 日付 | アプローチ | CV | LB |
| --- | --- | --- | --- |
| 2026-07-29 | ロジスティック回帰(基本特徴+Title+FamilySize) | 0.8283 | 0.76794 |
| 2026-07-29 | LightGBM(同上) | 0.8227 | - |
| 2026-07-29 | RandomForest(同上) | 0.8137 | - |
| 2026-07-29 | LightGBM(+Title別Age補完+Ticketグループ特徴) | 0.8339 | - |
| 2026-07-29 | LightGBM tuned + グループ運命連動特徴(lgbm_t/wcg) | 0.8541 | - |
| 2026-07-29 | 同上、集計を女性・子供に限定した本家WCG方式(lgbm_t/wcg2) | 0.8608 | 0.76794 |
| 2026-07-31 | 同上+姓ベース拡張グループ+cabin(lgbm_t/wcg2sc) | 0.843 (repeated) | **0.77751** |
| 2026-08-03 | logreg+rf+lgbm_t の確率平均(avg3/wcg2sc) | 0.8484 (repeated) | 0.75358(棄却) |
| 2026-08-03 | wcg2sc+IsWCフラグ(lgbm_t/wcg2scw) | 0.8455 (repeated) | 0.77272(棄却) |

これまでの実験の経緯と教訓は [docs/experiment-history.md](docs/experiment-history.md) にまとめている。

### 立ち位置(2026-08-03 時点の公開LB、11,194チーム)

- 自己ベスト 0.77751 は**上位 29.9%**(中央値 0.77511)。1人正解 = +0.239%。
- +2人で上位19%、+5人で上位10%、+10人(0.801)で上位5%。0.85 超は答えの暗記勢。
- 可視化は `notebooks/lb_position.py`(LB スナップショットの取得コマンドはファイル冒頭参照)。
- test 側の残り誤答の所在は `notebooks/test_gap_analysis.py` で推定
  (female C3 / male C1 / male C3 に集中)。
- CV と LB のデータ性質の差は `notebooks/train_test_shift.py`
  (adversarial validation AUC 0.500 = 属性分布は同一。WCG 証拠は train 29% vs test 27%、
  ただし female C3 のみ 42%→28% と test で薄い)。
- 難所層の正解行/不正解行の比較は `notebooks/hard_segment_eda.py`
  (正誤を分けるのは属性ではなく WCG 証拠の有無。証拠外の生死は属性同一人物間で分かれる=運)。
- Age 補完の検討は `notebooks/age_imputation_eda.py`
  (MAE は Title×Pclass・回帰で 9.4→8.6 歳まで改善するが survival は不動 → 現行 Title 別を維持)。

学習・評価は `src/train.py`(使い方はファイル冒頭の docstring 参照)。

## データ

```bash
kaggle competitions download -c titanic -p data
unzip -o data/titanic.zip -d data
```

- `train.csv` … 891 件(ラベルあり)
- `test.csv` … 418 件(予測対象)
- `gender_submission.csv` … 提出フォーマットのサンプル
