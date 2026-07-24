# CLAUDE.md

Kaggle コンペティション用リポジトリ。コンペごとに `competitions/<コンペ名>/` を切って管理する。

## ディレクトリ構成

- `competitions/<コンペ名>/data/` … kaggle CLI で取得(git 管理外)
- `competitions/<コンペ名>/notebooks/` … EDA・実験ノートブック
- `competitions/<コンペ名>/src/` … 前処理・学習・推論スクリプト
- `competitions/<コンペ名>/submissions/` … 提出ファイル(上書きせず履歴として残す)

## 実験ルール

- 乱数シードを固定し、CV 設定はコンペ内で統一する。
- submission ファイルは `submissions/` に日付・内容が分かる名前で残す。
- 各コンペの概要・アプローチ・結果はそのコンペの README.md に記録する。
