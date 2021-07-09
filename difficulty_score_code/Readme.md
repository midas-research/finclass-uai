# Difficulty Score Computation
Here we give the code for the Stock and Model Complexity presented in FinCLASS

## Contents

- `text_price_difficulty_stocknet.py` - The script for calculating the price, text, text+price complexity for the US S&P 500 data
- `text_price_difficulty_china.py` -  The script for calculating the price, text, text+price complexity for the China & Hong Kong data
- `model_difficulty.py` - The script for calculating the model complexity
- `cross_review.py` - The script for training the identical models on the meta datasets for calculating the model complexity, set the `bucket` parameter to i train on the ith bucket

- `model.py` - The model definition for calculating the model difficulty
- `dataset.py` - The dataset class

Some parameters worth noting here in various model scripts
  - `--model` -> [simple, time], choses the type of model, simple - without TLSTM, time - with TLSTM
  - `--data` -> [stock, china] The dataset being used
  - `--use_attn1` [true, false]-> If to use the intraday attention
  - `--use_attn2` [true, false]-> If to use the interday attention

## Steps to run
```
1. python3 text_price_difficult_stocknet.py
2. python3 cross_review.py --data stock/china --bucket [run for 0-Num_buckets]
3. python3 model_difficulty.py --data stock/china
```

## Outputs

Running the above scripts creates .pkl files containing the various complexity scores namely

- `price_difficulty.pkl` - The price only difficulty
- `text_difficulty.pkl` - The text only difficulty
- `text_price_difficulty.pkl` - The text + price difficulty ( the stock complexity )
- `model_difficulty.pkl` - The model complexity
