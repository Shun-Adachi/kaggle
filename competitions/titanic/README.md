# Titanic - Machine Learning from Disaster

https://www.kaggle.com/competitions/titanic

乗客の属性データから生存(Survived: 0/1)を予測する二値分類。評価指標は Accuracy。

## 結果

| 日付 | アプローチ | CV | LB |
| --- | --- | --- | --- |
| - | - | - | - |

## データ

```bash
kaggle competitions download -c titanic -p data
unzip -o data/titanic.zip -d data
```

- `train.csv` … 891 件(ラベルあり)
- `test.csv` … 418 件(予測対象)
- `gender_submission.csv` … 提出フォーマットのサンプル
