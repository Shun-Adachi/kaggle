# Kaggle Competitions

Kaggle コンペティションへの取り組みをまとめたリポジトリです。
コンペごとに `competitions/<コンペ名>/` ディレクトリを分けて管理しています。

## Competitions

| コンペ | 状況 | ベストスコア (CV / LB) |
| --- | --- | --- |
| [Titanic](competitions/titanic/) | 準備中 | - |

## セットアップ

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# データ取得(例: Titanic)
kaggle competitions download -c titanic -p competitions/titanic/data
unzip -o competitions/titanic/data/titanic.zip -d competitions/titanic/data
```

## ディレクトリ構成(コンペ共通)

```
competitions/<コンペ名>/
├── data/         # コンペデータ(git 管理外)
├── notebooks/    # EDA・実験ノートブック
├── src/          # 前処理・学習・推論スクリプト
└── submissions/  # 提出ファイルの履歴
```
