# Small Auto Chess

## 概要

このプロジェクトは、PythonのPygameとKerasを使用して開発された自動チェスゲームです。プレイヤーが白駒を操作してAIと対戦できるモードや、AI同士が自動で対戦して結果を出力するモードがあります。加えて、AIモデルを自作してゲームに反映させることも可能です。
またこのプログラムは改良,修正する場合があります

## 特徴

* **チェスAI対戦**: プレイヤーは`chess12.py`を使用してAIと対戦することができます。
* **AI同士の自動対戦**: `automode.py` では、AI同士の対戦を自動で行い、結果を`json`ファイルに出力します。
* **AIモデルのカスタマイズ**: `black_model.py`を使用して新しいAIモデルを作成し、任意のモデルを使用することが可能です。
* **Kerasモデルの使用**: `small_chess_totsuka.keras`（攻撃用）と `small_chess_yata.keras`（防御用）という事前トレーニングされたモデルを使用しています。

## プログラムの主要ポイント

1. **二つのモデルの使い分け**:
   - 攻撃用と防御用の2つのモデルを使用することで、チェスAIの「もじもじ現象」（同じ場所を行ったり来たりする現象）を解消しています。

2. **高度な盤面評価**:
   - 現在の盤面だけでなく、次の次の手まで予測して、最も有利になりそうな手を選択します。(厳密には盤面スコア、移動したときにとれる駒スコアが最大になる場所を予測しています）

3. **Minmax法との比較**:
   - 従来のminmax法では、先読みする手数が多いほど強くなる傾向があり、同じ手数の場合は後攻が不利（6割以上の確率で負ける）でした。（イメージ的には　黒2<黒3≦白2<白3です）先攻の場合は一つ上のレベルに勝てることもありますが、の場合は大体勝てません.
   - 本プログラムでは、学習によってこの問題を改善し、より公平な対戦を実現しています。

4. **学習による性能向上**:
   - 学習を重ねることで、AIの性能が向上します。
   - 例：ミニマックスアルゴリズムで黒が3手先読み、白が2手先読みした対戦で学習したモデルは、ミニマックスアルゴリズムで対決していた時と比べ比較して白2手に対する勝率が向上します。さらに3手先読みした場合にも勝つことができます.
  
     ## ファイル構成

```
.
├── asset/
│   ├── BBishop.png
│   ├── BKing.png
│   ├── BKnight.png
│   ├── BPawn.png
│   ├── BQueen.png
│   ├── BRook.png
│   ├── WBishop.png
│   ├── WKing.png
│   ├── WKnight.png
│   ├── WPawn.png
│   ├── WQueen.png
│   ├── WRook.png
├── ai_chess_ver6.json    # 自動対戦結果を保存するファイル
├── level5_chess_ver6.json    # 自動対戦レベル5の結果を保存するファイル
├── small_chess_totsuka.keras    # AIの攻撃モデル
├── small_chess_yata.keras    # AIの防御モデル
├── automode.py    # AI同士の自動対戦モード
├── black_model.py    # 新しいAIモデルを作成するスクリプト
├── chess12.py    # プレイヤーが白駒を操作してAIと対戦するモード
└── smallchess_6.py    # 自動対戦のメインプログラム
```

## 必要条件

以下のライブラリが必要です：

* `pygame`
* `tensorflow` または `keras`
* `numpy`
* `random`
* `copy`

インストールは次のコマンドで行います：

```bash
pip install pygame tensorflow numpy
```

## 使い方

1. プロジェクトのクローン

   このリポジトリをローカルにクローンします。

   ```bash
   git clone https://github.com/yourusername/small-auto-chess.git
   cd small-auto-chess
   ```

2. ピースの画像を配置

   `asset`フォルダ内に、各チェス駒の画像を配置します。以下の名前で画像を保存してください。

   * `BPawn.png` （黒のポーン）
   * `BRook.png` （黒のルーク）
   * `BKnight.png` （黒のナイト）
   * `BBishop.png` （黒のビショップ）
   * `BQueen.png` （黒のクイーン）
   * `BKing.png` （黒のキング）
   * `WPawn.png` （白のポーン）
   * `WRook.png` （白のルーク）
   * `WKnight.png` （白のナイト）
   * `WBishop.png` （白のビショップ）
   * `WQueen.png` （白のクイーン）
   * `WKing.png` （白のキング）

3. モデルの使用

   デフォルトでは、`small_chess_totsuka.keras`（攻撃用）と`small_chess_yata.keras`（防御用）のモデルを使用します。これらのモデルはすでにプロジェクトに含まれています。

   ### 新しいモデルの使用

   自分で新しいAIモデルを作成したい場合は、`black_model.py`を使用して新しいモデルをトレーニングできます。その後、`chess12.py`や`automode.py`で読み込むモデルを新しいものに変更してください。

4. ゲームの起動

   ### プレイヤー vs AI

   白駒を操作してAIと対戦するには、以下のコマンドを実行します。

   ```bash
   python chess12.py
   ```

   ### AI同士の自動対戦

   AI同士の対戦を自動で実行するには、以下のコマンドを実行します。

   ```bash
   python automode.py
   ```

## ファイルごとの説明

### `chess12.py`

プレイヤーが白駒を操作し、黒駒を操作するAIと対戦するモードです。クリック操作で駒を移動させ、ゲームが進行します。

### `automode.py`

AI同士を自動で対戦させ、その結果を`ai_chess_ver6.json`や`level5_chess_ver6.json`ファイルに保存します。

### `black_model.py`

新しいAIモデルを作成するためのスクリプトです。Kerasを使用してモデルをトレーニングし、保存します。

### `smallchess_6.py`

自動対戦モードの本体です。このファイルでAIが黒駒・白駒を操作し、チェスの対戦を自動で行います。
