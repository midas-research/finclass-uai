## Model Training
Here we give the code to train the THA-Net model with and without FinCLASS

## Contents

- `train_without_curriculum.py` - The script to run THA-Net with FinCLASS
- `train_curriculum.py` - The script to run THA-Net without FinCLASS
- `model.py` - The model definition for calculating the model difficulty
- `dataset.py` - The dataset class

Some parameters worth noting here in various model scripts
  - `--model` -> [simple, time], choses the type of model, simple - without TLSTM, time - with TLSTM
  - `--data` -> [stock, china] The dataset being used
  - `--task` -> [movement, volatility]  Task on which to train model 
  - `--use_attn1` [true, false]-> If to use the intraday attention
  - `--use_attn2` [true, false]-> If to use the interday attention
  - `--num_epochs` -> Number of epochs to train for 
  - `--batch_size` -> Batch size to use


## Steps to Train
To run THA-Net with FinCLASS
```
python3 train_with_curriculum.py --data stock/china --task movement/volatility --use_attn1 True --use_attn2 True
```

To run THA-Net without FinCLASS
```
python3 train_without_curriculum.py --data stock/china --task movement/volatility --use_attn1 True --use_attn2 True
```


