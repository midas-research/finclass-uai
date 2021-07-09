# Dataset Preprocessing

Here we give the code for preprocessing the datasets used for the format to be used in the model.

## Data

Find the US S&P 500 data [here](https://github.com/yumoxu/stocknet-dataset), and the China & Hong Kong data [here](https://pan.baidu.com/s/1mhCLJJi). Please extract the corresponding dataset, and specify the path to the tweets and price data in the `path_tweet_data` and `path_price_data` variables in the below scripts.

## Contents

- `data_process_stocknet.py` - Script to process the US S&P 500 dataset
- `gen_train_test_split_stocknet.py` - Script to split the US S&P 500 dataset into train-test sets
- `data_process_china.py` - - Script to process the China & Hong Kong dataset
- `gen_train_test_split_china.py` - Script to split the China & Hong Kong dataset into train-test sets

## Steps to run

For processing US S&P 500 dataset run
```
1. python3 data_process_stocknet.py 
2. python3 gen_train_test_split_stocknet.py
```
For processing China & Hong Kong dataset run
```
1. python3 data_process_china.py 
2. python3 gen_train_test_split_china.py
```

## Processed Data

The above scripts create .pkl files containg train-test data for both the datasets. With each file consisting of the following for sample input into the FinCLASS pipeline

- `embedding` - Text embeddings for each day in the lookback, we use the 768-dimensional BERT embedding for each tweet/news in the dataset, by averaging the token-level outputs from the final layer of BERT.
- `time_feature` - The time feature, calculated by taking the inverse of the interval between texts in minutes.
- `price_feature` - The normalized HLC prices for each stock ( used in price entropy calculation )
- `movement_label` - The label for the stock movement prediction task ( 1 means price will go up and 0 otherwise )
- `volatility` - The target for the volatility regression task, ie. the single day log volatility for the target date
  