## 概要  
ゼミで Transformersやlm-evalなどのライブラリの使い方を解説した際に使用したコード集です。

## バージョン / Version

python 3.9

## 使用方法 / Usage

[scripts](scripts/) 内に以下の Jupyter Notebook ファイルが含まれています：

- [pipeline.ipynb](scripts/pipeline.ipynb)：  
  Transformers の `pipeline`を使ったシンプルなテキスト分類などの実装例。
  
- [BERT.ipynb](scripts/BERT.ipynb)：  
  BERT モデルの基本的な使い方（事前学習済みモデルの読み込み、ファインチューニングの概要など）。
  
- [lm-eval.ipynb](scripts/lm-eval.ipynb)：  
  大規模言語モデルの評価。

- [GRPO.ipynb](scripts/GRPO.ipynb)：  
  deepseek R1などの推論モデルを支える技術。

$$ J_{\text{GRPO}}(\theta) =\frac{1}{G}\sum_{i=1}^{G} {\min\!\left(\displaystyle\frac{\pi_\theta(o_i\mid q)}{\pi_{\text{old}}(o_i\mid q)}\,A_i \,, \; \text{clip}\left(\displaystyle\frac{\pi_\theta(o_i\mid q)}{\pi_{\text{old}}(o_i\mid q)},1-\varepsilon,1+\varepsilon\right)\,A_i \right)} -\beta D_{\mathrm{KL}}\!\bigl(\pi_\theta\ \|\ \pi_{\text{ref}}\bigr) $$