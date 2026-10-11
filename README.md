# 楕円曲線・保型形式・L関数のPython実験

有理点の群演算から有限体の点数、保型形式のFourier係数、L関数の数値評価までを、数式と実験でつなぐ。定義、導出、具体的な入力、結果の読み方は `analysis.ipynb` と `qiita_article.md` に記載している。

## ファイル

```text
elliptic-modular-lfunctions/
├── analysis.ipynb
├── module.py
├── README.md
└── requirements.txt
```

ノートブックは一つだけで、共通処理は `module.py` から読み込む。図17点はノートブックの出力データに埋め込まれているため、別の画像フォルダは不要。係数や局所データの入力はコード内に定義してあり、パッケージ導入後の主要解析はネットワークにアクセスしない。

## 実行

Python 3.13.5 / Linuxで動作確認した。パッケージは `requirements.txt` の版を使う。異なるPythonやパッケージ版については未検証。

```bash
cd elliptic-modular-lfunctions
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m ipykernel install --user \
  --name elliptic-lab --display-name "Python (elliptic-lab)"
jupyter lab analysis.ipynb
```

Jupyterで `Python (elliptic-lab)` を選び、カーネルを再起動して全セルを実行する。必ず `module.py` と同じディレクトリで実行する。既存の実行済み出力は、そのまま読んでもよい。

主要な回帰チェックだけなら次を使う。

```bash
python module.py
```

GUIを使わずノートブックを再実行するには、同じディレクトリで次を実行する。`--inplace` はノートブック内の実行結果を更新する。

```bash
jupyter nbconvert --to notebook --execute --inplace analysis.ipynb \
  --ExecutePreprocessor.timeout=180
```

## 実行確認

全118セルのうちコード57セルを先頭から実行し、エラーは0。366件の回帰チェックが `PASS`。記事中のPythonコード例36個も掲載順に別途実行して確認した。図は17点。所要時間はこの環境で約22秒だったが、CPU、BLAS、インポート、パッケージ版によって変わる。

PARI/GPによる任意の追加照合は、実行環境に `gp` がないため `SKIP`。主要解析には不要であり、未実行の経路を検証済みと扱わない。`gp` が存在する環境では導手、符号、係数、ランクの上下限の追加照合を行う。

## 主な内容

|節|内容|
|:--|:--|
|0|記号、前提知識、厳密な演算・有限検査・数値近似の区別|
|1|合同数と有理三角形、Tunnellの必要条件、平方和、Dirichlet指標|
|2|Weierstrass不変量、一般形の群演算、15種類のtorsionの完全性、還元、Hasse境界、高さ|
|3|平方類写像、補間による半分の構成、合同数34の64候補の完全分類、p進数、SelmerとSha|
|4|複素格子、Weierstrass関数、正方格子の周期、基本領域、保型曲線の種数|
|5|Eisenstein級数、Δ、j、η積、400係数の独立照合、Sturm境界、Hecke行列、Petersson積|
|6|Euler因子、L係数の再帰、保型性の主張、Mellin変換、複素値、中心値・中心微分と尾の上界|
|7|ランク0〜3のL関数、中心展開、BSD公式、周期、合同数34・82・157|
|8|Frey曲線、複素指数、射影空間、Hensel持ち上げ、Tate標準形|
|9–10|規約の検査、テスト、API、各結論の意味|

## 計算結果の位置づけ

整数・分数の群演算、有限体の直接計数、指定次数までの形式的級数は厳密に計算する。合同数34のランク2は、降下法の定理と64候補の完全分類から導く。15種類のtorsion例は、生成部分群だけでなくMazurの分類と良い還元の単射を使って全位数も確認する。

高さ、格子和、Petersson積、Mellin求積は近似。数値的なゼロや行列式から、保証なしに独立性・解析的ランクを証明したとは扱わない。中心値の尾の上界は丸め誤差を含まない。

Mordell–Weil、Mazur、降下写像の核、保型性などの大きな定理は、使用する内容を解説内で明示したうえで利用する。任意の曲線の最小化、Tateアルゴリズム、一般のSelmer群・Sha・ランクの完全計算は実装していない。既知の群構造を入力する計算と、数値からBSDの量を逆算する計算は別扱いにしている。

## APIの短い使用例

```python
from fractions import Fraction as F
import module as lab

E = lab.EllipticCurve(a4=-25)
P = E.point(-4, 6)
assert E.mul(2, P) == E.point(F(1681, 144), F(-62279, 1728))
assert E.add(P, E.neg(P)) is None

coeff_ec = lab.l_coefficients(lab.CURVES["11a1"], 400)
coeff_eta = lab.eta_product_coefficients({1: 2, 11: 2}, 400)
assert list(coeff_ec) == coeff_eta

result = lab.central_L(lab.CURVES["11a1"], N=160, dps=60)
print(result)
```

`None` は無限遠点。分数は `F(1, 10)` と入力し、厳密な座標へ `0.1` を渡さない。`central_L` の引数 `N` は導手ではなく項数。各APIの入力、戻り値、制約は解説末尾に一覧を置いた。

