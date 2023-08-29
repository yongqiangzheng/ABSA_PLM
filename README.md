# NLP

## Run

1. download pre-trained models and process dataset

```
python preprocess.py
```

2. fine-tune models

You can fine-tune single model with:

```
python finetune.py --model_name xxx --dataset xxx --lr xxx
```

or fine-tune all models:

```
bash run.sh
```

3. evaluate results and web demo(not finished!)

```
#python demo.py
```

## Results

top: 3-runs average

bottom: 3-runs best

| Model        | Lap14            |                  | Rest14           |                  | MAMS             |                  |
| ------------ | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- | ---------------- |
|              | Acc(%)           | F1(%)            | Acc(%)           | F1(%)            | Acc(%)           | F1(%)            |
| BERT         | 78.63<br />78.84 | 74.72<br />75.01 | 85.33<br />85.62 | 78.44<br />79.26 | 83.26<br />83.61 | 82.75<br />83.20 |
| RoBERTa      | 81.77<br />81.97 | 78.31<br />78.50 | 87.71<br />88.30 | 81.64<br />82.53 | 83.16<br />84.06 | 82.46<br />83.44 |
| BERT_LoRA    | 77.74<br />78.84 | 72.73<br />74.98 | 84.70<br />85.00 | 76.84<br />77.24 | 82.44<br />83.23 | 81.67<br />82.58 |
| RoBERTa_LoRA | 82.24<br />82.76 | 79.27<br />79.97 | 87.05<br />87.59 | 80.10<br />81.05 | 84.03<br />84.43 | 83.47<br />84.03 |
