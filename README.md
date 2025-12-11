# Jasmine_orbit

developer: Yui Kasagi

## Introduction

ターゲットの visibility チェック用ツール

基本的には RPR-SJ4B0509 (by H. Kataza) を流用しています。

非観測時の姿勢は RPR-SJ512017B を参照し、`OrbitAttitude.py` の対応する部分を修正しました。

## Usage

### 設定

各種閾値、アウトプットディレクトリへのパスは `src/jasmine_orbit/settings.py` で管理しています。
適宜書き換えて使用してください。

### 放射板への熱入力量の推測

例）

春分 (-s) から 45日後 (-p 45.0) を開始日として、90日間 (-w 90.) GJ 3929 (-t GJ 3929) の計算を実施。結果を図・データとしてアウトプットする(-o)。

```
python main_target.py -s -p 45.0 -w 90. -o -t GJ 3929 
```

オプション解説：

```
usage:
    main_target.py [-h|--help] (-s|-a) -p <day_offset> -w <days> [-o] [-t <target_name>] [-m <minutes>]

options:
    -h --help       show this help message and exit
    -s              春分点を基準
    -a              秋分点を基準
    -p <day_offset> 基準日からの計算開始日(この日を含む)
    -w <days>       計算期間(日)
    -o              グラフ出力(True or False)
    -t <target_name>    target name
    -m <minutes>   time step in minutes [default: 1]
```