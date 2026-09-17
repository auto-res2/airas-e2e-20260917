# 仮説と実験設計（ドラフト、record 未登録）

## 読んだ文献（`preregister_record` に渡す順、s1 … s4）

| id | 論文 | 識別子 | fulltext_path |
| --- | --- | --- | --- |
| s1 | Müller, Kornblith, Hinton. When Does Label Smoothing Help? NeurIPS 2019 | arXiv 1906.02629 / DOI 10.48550/arxiv.1906.02629 | ~/.airas/cache/fulltext/f5927a63479a4f6b0d9a42054d15052d59a1d265556f2f64236d48a88feae1bc.txt |
| s2 | Chandrasegaran, Tran, Zhao, Cheung. Revisiting Label Smoothing and Knowledge Distillation Compatibility: What was Missing? ICML 2022 | airas_db 1437751a77305a0c4c8d44127bd8c285 / https://proceedings.mlr.press/v162/chandrasegaran22a/chandrasegaran22a.pdf | ~/.airas/cache/fulltext/ea21da839bb408bfba9ee5e3c35de1eaaa1463822b865d2ae1a3d491c8b2299d.txt |
| s3 | Sultan. Knowledge Distillation ≈ Label Smoothing: Fact or Fallacy? EMNLP 2023 | airas_db 271 / https://aclanthology.org/2023.emnlp-main.271.pdf | ~/.airas/cache/fulltext/0d14d67658890442f2175b3f0f1394ae29a3489d9f938e9523aa3d51a42f4e88.txt |
| s4 | Shen et al. Is Label Smoothing Truly Incompatible with Knowledge Distillation: An Empirical Study. ICLR 2021 | airas_db d64a340bcb633f536d56e51874281454 / arXiv 2104.00676 | ~/.airas/cache/fulltext/5b757e682b352ad177c3c76641a7984e6b120b04239f596b20994ff98013dc6d.txt |

### 引用箇所（page、逐語）

- s1.p1 (p.1): "if a teacher network is trained with label smoothing, knowledge distillation into a student network is much less effective"
- s1.p2 (p.4): "we show that label smoothing also reduces ECE and can be used to calibrate a network without the need for temperature scaling"
- s2.p1 (p.1): "This systematic diffusion essentially curtails the benefits (as claimed by Shen et al. (2021b)) obtained by distilling from an LS-trained teacher, thereby rendering KD at increased temperatures ineffective"
- s2.p2 (p.2): "We suggest to use an LS-trained teacher with a low-temperature transfer (i.e. T = 1) to achieve high performance students"
- s3.p1 (p.1): "In KD, the student inherits not only its knowledge but also its confidence from the teacher"
- s3.p2 (p.4): "As a student learns to mimic its teacher in KD, we see in the above results that it also inherits its confidence"
- s4.p1 (p.1): "label smoothing erases relative information between teacher logits"

## ギャップ

LS と KD の相性は精度で議論されてきた（s1 は不利、s4 は反論、s2 が温度で調停）。s2 は ImageNet/CUB/NMT の大規模実験を行うが、較正（ECE）を一度も測っていない。一方 s3 はテキスト分類で「生徒は教師の confidence を継承する」と示したが、画像分類・LS 教師では未確認。s1 が示した LS の較正効果が、s2 の推奨設定（LS 教師、T=1）で生徒に**継承されるか**は、どの論文も答えていない。

## 仮説 h1

LS で学習した教師から蒸留した生徒は、hard target で学習した教師から蒸留した生徒より較正誤差（ECE）が低く、この差は s2 が精度の観点で推奨する低温度（T=1）でも、高温度（T=4）でも現れる。すなわち s3 の「confidence の継承」は画像分類の LS 教師にも成り立ち、s1 の較正効果は蒸留を通じて生徒に伝わる。

- grounded_on: s1.p1, s1.p2, s2.p1, s2.p2, s3.p1
- 反証条件: T=1 で LS 教師の生徒の ECE が hard 教師の生徒より 0.01 以上低くならなければ h1 は反証される（c1）。

## 計算環境（要確認）

- 想定: NVIDIA GPU 1 枚（T4/A10 級で十分）、arch `x86_64`。**ユーザー未確認の仮定**。
- CIFAR-100 で ResNet-18 教師 + 小型生徒、各 run 100 epoch 以内。

## 実験設計

- データセット: CIFAR-100（curated: https://huggingface.co/datasets/uoft-cs/cifar100、train 50k / test 10k、100 クラス）。較正は test 10k で 15-bin ECE。
- 教師: ResNet-18（CIFAR 版）。生徒: ResNet-8 相当の小型 CNN（config で固定）。
- LS 係数 α=0.1（s1 の標準設定）。KD 損失: α_kd=0.9 の KL(T) + 0.1 CE、温度 T∈{1, 4}。
- メトリクス（英語識別子、`<run_id>.<metric>` で参照）: `accuracy`, `ece`, `nll`。
- seed は config で固定 1 本（seed 数は将来 append で拡張可、run_id は変えない）。

### run 一覧（run は 1 claim にのみ属する）

| run_id | 内容 | params |
| --- | --- | --- |
| teacher-hard | ResNet-18, CE（α=0） | mode=full |
| teacher-ls | ResNet-18, LS α=0.1 | mode=full |
| student-kd-hard-t1 | 生徒、teacher-hard から T=1 で蒸留 | mode=full |
| student-kd-ls-t1 | 生徒、teacher-ls から T=1 で蒸留 | mode=full |
| student-kd-hard-t4 | 生徒、teacher-hard から T=4 で蒸留 | mode=full |
| student-kd-ls-t4 | 生徒、teacher-ls から T=4 で蒸留 | mode=full |

### claims（criterion は `(subject.metric - reference) op margin`）

- **c1** 生徒の ECE 継承（T=1、主張の中核）: `student-kd-ls-t1.ece - student-kd-hard-t1.ece <= -0.01`。prediction [-0.06, -0.01]、basis: s1 が報告する LS による ECE 低減の一部が生徒に残る想定。cites: s2.p2, s3.p1, s3.p2。rationale: s2 の推奨設定で較正が継承されれば h1 の主節が成り立つ。runs: student-kd-hard-t1, student-kd-ls-t1。
- **c2** 教師側の再現: `teacher-ls.ece - teacher-hard.ece <= -0.02`。prediction [-0.10, -0.02]、basis: s1 Table 3 の傾向。cites: s1.p2。rationale: 教師に較正差がなければ「継承」は検証不能。runs: teacher-hard, teacher-ls。
- **c3** 高温度でも継承: `student-kd-ls-t4.ece - student-kd-hard-t4.ece <= -0.01`。prediction [-0.06, -0.005]、basis: s2 は T が高いと精度の利得が消えるというが、confidence の継承（s3）は温度と独立と予想。cites: s2.p1, s3.p1。rationale: h1 の「T=4 でも現れる」の部分。runs: student-kd-hard-t4, student-kd-ls-t4。

### assumptions（h1 に到達するために claims の外で認めるもの）

- 15-bin ECE が較正の代表指標である（c1–c3）。
- CIFAR-100 と ResNet-18/小型 CNN の 1 seed の結果が他のデータセット・規模に一般化する（c1–c3）。
- `accuracy` は表で報告するが criterion にしていない。LS 教師の生徒の精度が大きく劣らないこと（s2 の推奨の前提）は本研究では測るだけで検証しない（c1, c3）。

## 次のステップ

`preregister_record(literature=s1..s4 + passages, hypotheses=[h1])` で freeze commit を作り、`.research/latex/mdpi/main.tex` を書いて `verify_latex` が通るまで直す。
